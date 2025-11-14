#!/usr/bin/env python3
from fastapi import FastAPI, Request
from fastapi.background import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from Bot import bot
from Bot.utils import sendImagesToUser
from telebot.types import Update
from config import WEBHOOK_PATH, WEBHOOK_URL, SAVE_DIR
from models import ImageData, ImageProcessResponse
from ai import process_image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    json_data = await request.json()
    update = Update.de_json(json_data)
    bot.process_new_updates([update])
    return {"ok": True}

@app.post('/upload-image')
async def upload_image(data: ImageData, background_tasks: BackgroundTasks) -> ImageProcessResponse:
    '''header, b64 = data.img.split(',', 1)
    ext = ".png" if "png" in header else ".jpg"
    raw = base64.b64decode(b64)
    filename = os.path.join(SAVE_DIR, f"{uuid.uuid4().hex}{ext}")

    with open(filename, "wb") as f:
        f.write(raw)

    return {"ok": True, "filename": filename}
    '''
    
    images = await process_image(data)
    background_tasks.add_task(
        sendImagesToUser,
        bot,
        data.user_id,
        data.image,
        images
    )

    return images


@app.get('/')
async def index():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    bot.delete_webhook()
    bot.set_webhook(WEBHOOK_URL)
    uvicorn.run(app, host="0.0.0.0", port=8000)
