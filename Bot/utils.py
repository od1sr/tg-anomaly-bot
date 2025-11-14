from telebot import TeleBot
from telebot.types import InputMediaPhoto
from models import ImageProcessResponse
import base64

def _decode_base64_image(b64_string: str) -> bytes:
    """Декодирует строку base64 в байты, удаляя префикс, если он есть."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    return base64.b64decode(b64_string)

def sendImagesToUser(bot: TeleBot, user_id: int, original_img: str, images: ImageProcessResponse):
    '''
    Отправляет пользователю исходное изображение и изображения, сгенерированные ИИ.

    args:
      - *bot*: TeleBot instance
      - *user_id*: int, user id
      - *original_img*: str, base64
      - *images*: models.ImageProcessResponse
    '''
    try:
        original_bytes = _decode_base64_image(original_img)
        cleaned_bytes = _decode_base64_image(images.cleaned)
        heatmap_bytes = _decode_base64_image(images.heatmap)
        overlay_bytes = _decode_base64_image(images.overlay)

        media = [
            InputMediaPhoto(original_bytes),
            InputMediaPhoto(cleaned_bytes, caption="Очищенное"),
            InputMediaPhoto(heatmap_bytes, caption="Тепловая карта"),
            InputMediaPhoto(overlay_bytes, caption="Наложение")
        ]

        bot.send_media_group(chat_id=user_id, media=media)
    except Exception as e:
        bot.send_message(user_id, f"Произошла ошибка при отправке изображений: {e}")