from config import BOT_TOKEN
import telebot
from .handlers import register_handlers

bot = telebot.TeleBot(BOT_TOKEN)
register_handlers(bot)