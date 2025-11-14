from pydantic import BaseModel
from typing import Optional

class ImageData(BaseModel):
    image: str  # data:image/png;base64,...
    user_id: Optional[int] = None  # optional, can be int or None
    return_heatmaps: Optional[bool] = True
    heatmap_alpha: Optional[float] = 0.5

class ImageProcessResponse(BaseModel):
    success: bool
    cleaned: Optional[str] = None  # base64 очищенного изображения
    heatmap: Optional[str] = None  # base64 тепловой карты
    overlay: Optional[str] = None  # base64 наложения
    original_size: Optional[list] = None
    error: Optional[str] = None