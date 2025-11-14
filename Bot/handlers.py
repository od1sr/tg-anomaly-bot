import telebot
from config import WELCOME_MESSAGE

def register_handlers(bot: telebot.TeleBot):

    @bot.message_handler(func=lambda m: True)
    def echo_all(message):
        bot.reply_to(message, WELCOME_MESSAGE)
