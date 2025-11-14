import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import cv2
import base64
import io
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from models import ImageData, ImageProcessResponse

IMG_SIZE = 64

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.ModuleList([nn.MaxPool2d(2), DoubleConv(64, 128)])
        self.down2 = nn.ModuleList([nn.MaxPool2d(2), DoubleConv(128, 256)])
        self.down3 = nn.ModuleList([nn.MaxPool2d(2), DoubleConv(256, 512)])
        self.up1 = nn.ModuleList([nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), DoubleConv(512, 256)])
        self.up2 = nn.ModuleList([nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), DoubleConv(256, 128)])
        self.up3 = nn.ModuleList([nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), DoubleConv(128, 64)])
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1[0](x1)
        x2 = self.down1[1](x2)
        x3 = self.down2[0](x2)
        x3 = self.down2[1](x3)
        x4 = self.down3[0](x3)
        x4 = self.down3[1](x4)

        x = self.up1[0](x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up1[1](x)

        x = self.up2[0](x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2[1](x)

        x = self.up3[0](x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3[1](x)

        logits = self.outc(x)
        return self.final_activation(logits)

# Загрузка модели
DEVICE = "cpu"
MODEL = UNet(n_channels=3, n_classes=3).to(DEVICE)
MODEL.load_state_dict(torch.load("ModelNeurons.pth", map_location=torch.device(DEVICE)))

# FastAPI приложение
app = FastAPI(
    title="Anomaly Detection API",
    description="API для обнаружения и удаления аномалий на изображениях",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def base64_to_image(base64_string: str) -> Image.Image:
    """
    Конвертирует base64 строку в PIL Image
    """
    try:
        # Убираем префикс если есть (data:image/png;base64,)
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Декодируем base64
        image_data = base64.b64decode(base64_string)
        
        # Создаем PIL Image из байтов
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        return image
    except Exception as e:
        raise ValueError(f"Ошибка декодирования base64: {e}")


def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """
    Конвертирует PIL Image в base64 строку
    """
    try:
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return base64_string
    except Exception as e:
        raise ValueError(f"Ошибка кодирования в base64: {e}")

def create_heatmap(anomaly_map: np.ndarray, original_size: tuple, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Создает тепловую карту из карты аномалий
    """
    # Нормализуем карту аномалий к [0, 255]
    anomaly_normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
    anomaly_normalized = anomaly_normalized.astype(np.uint8)
    
    # Применяем цветовую карту
    heatmap = cv2.applyColorMap(anomaly_normalized, colormap)
    
    # Конвертируем из BGR в RGB
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Возвращаем к оригинальному размеру
    heatmap_resized = cv2.resize(heatmap_rgb, original_size)
    
    return heatmap_resized

def create_overlay(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Создает наложение тепловой карты на оригинальное изображение
    """
    # Убеждаемся, что оба изображения в uint8
    original_uint8 = original_image.astype(np.uint8)
    heatmap_uint8 = heatmap.astype(np.uint8)
    
    # Создаем наложение
    overlay = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_uint8, alpha, 0)
    
    return overlay

async def process_image(request: ImageData) -> ImageProcessResponse:
    """
    Обработка изображения через нейросеть
    
    - **image**: base64 строка с изображением
    - **return_heatmaps**: возвращать ли тепловые карты (по умолчанию True)
    - **heatmap_alpha**: прозрачность тепловой карты (0.0 - 1.0)
    """
    try:
        # Конвертируем base64 в изображение
        original_image = base64_to_image(request.image)
        original_size = original_image.size
        original_np = np.array(original_image)
        
        # Ресайз до размера модели
        img_resized = original_image.resize((IMG_SIZE, IMG_SIZE))
        
        # Преобразование в тензор
        img_tensor = ToTensor()(img_resized).unsqueeze(0).to(DEVICE)
        
        # Обработка моделью
        MODEL.eval()
        with torch.no_grad():
            output = MODEL(img_tensor)
        
        # Вычисляем карту аномалий
        anomaly_map = torch.abs(img_tensor - output)
        
        # Подготовка результата
        output_np = output.squeeze().cpu().permute(1, 2, 0).numpy()
        output_np = (output_np * 255).astype(np.uint8)
        
        # Подготовка карты аномалий
        anomaly_np = anomaly_map.squeeze().cpu().permute(1, 2, 0).numpy()
        anomaly_single_channel = np.mean(anomaly_np, axis=2)
        
        # Возврат к оригинальному размеру
        output_original_size = cv2.resize(output_np, original_size)
        anomaly_original_size = cv2.resize(anomaly_single_channel, original_size)
        
        # Конвертируем результаты в base64
        response_data = {
            'success': True,
            'cleaned': image_to_base64(Image.fromarray(output_original_size)),
            'original_size': [original_size[0], original_size[1]]
        }
        
        # Добавляем тепловые карты если нужно
        if request.return_heatmaps:
            heatmap = create_heatmap(anomaly_original_size, original_size)
            overlay = create_overlay(original_np, heatmap, alpha=request.heatmap_alpha)
            
            response_data['heatmap'] = image_to_base64(Image.fromarray(heatmap))
            response_data['overlay'] = image_to_base64(Image.fromarray(overlay))
        
        return ImageProcessResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Ошибка обработки изображения: {str(e)}"
        )


async def process_image_simple(request: ImageData):
    """
    Упрощенная обработка изображения (только очищенное изображение)
    """
    # Переопределяем параметр return_heatmaps
    request.return_heatmaps = False
    return await process_image(request)

