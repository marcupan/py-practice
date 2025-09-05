import asyncio
import logging
import sys

# Імпорти з бібліотеки aiogram
from aiogram import Bot, Dispatcher, types, F  # F - для магічних фільтрів
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

# from aiogram.utils.markdown import hbold # Допоміжна функція для форматування (опціонально)

# --- Конфігурація ---
# ВАЖЛИВО: НІКОЛИ не зберігайте токен прямо в коді для реальних проектів!
# Краще використовувати змінні середовища або файли конфігурації.
BOT_TOKEN = "YOUR_BOT_TOKEN"  # !!! ЗАМІНІТЬ НА ВАШ СПРАВЖНІЙ ТОКЕН !!!

# --- Ініціалізація Бота та Диспетчера ---
# Створюємо екземпляр Бота. parse_mode=ParseMode.HTML дозволяє використовувати HTML-теги для форматування тексту.
bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
# Створюємо екземпляр Диспетчера. Він відповідає за обробку повідомлень.
dp = Dispatcher()


# --- Обробники команд та повідомлень (Handlers) ---

# Обробник для команди /start
@dp.message(CommandStart())
async def handle_start(message: Message):
    """
    Цей обробник реагує на команду /start
    """
    # Отримуємо ім'я користувача для привітання
    user_name = message.from_user.full_name
    # Відповідаємо користувачу. Використовуємо HTML теги для форматування.
    await message.answer(
        f"Привіт, <b>{user_name}</b>! 👋\nЯ простий ехо-бот на aiogram 3.\nНадішли /help, щоб дізнатись більше.")


# Обробник для команди /help
@dp.message(Command("help"))
async def handle_help(message: Message):
    """
    Цей обробник реагує на команду /help
    """
    # Формуємо текст довідки
    help_text = (
        "<b>Доступні команди:</b>\n"
        "/start - Перезапустити бота\n"
        "/help - Показати це повідомлення\n\n"
        "Просто надішли мені будь-яке текстове повідомлення, і я його повторю."
    )
    await message.answer(help_text)


# Обробник для всіх інших текстових повідомлень (має бути в кінці)
# F.text - це "магічний фільтр", який спрацьовує, якщо в повідомленні є текст
@dp.message(F.text)
async def echo_text_message(message: Message):
    """
    Цей обробник ловить будь-які текстові повідомлення, що не є командами,
    визначеними вище, і відправляє їх назад користувачеві.
    """
    try:
        # Відповідаємо тим же текстом
        await message.answer(f"Ви написали: {message.text}")
    except Exception as e:
        # Логуємо помилку, якщо щось пішло не так
        logging.error(f"Помилка в обробнику ехо: {e}")
        await message.answer("Ой, сталася помилка при обробці вашого повідомлення.")


# Обробник для повідомлень без тексту (наприклад, фото, стікери)
# Цей обробник має йти після обробника текстових повідомлень
@dp.message()
async def echo_other_types(message: Message):
    """
    Обробник для нетекстових повідомлень.
    """
    await message.answer("Я отримав ваше повідомлення, але вмію повторювати лише текст.")


# --- Головна функція для запуску бота ---
async def main() -> None:
    """
    Асинхронна функція запуску бота та обробки повідомлень.
    """
    # Починаємо процес отримання оновлень від Telegram (polling)
    # skip_updates=True - ігнорувати повідомлення, що накопичились, поки бот був вимкнений
    print("Запускаю бота...")
    await dp.start_polling(bot, skip_updates=True)


# --- Точка входу в програму ---
if __name__ == "__main__":
    # Налаштовуємо базове логування, щоб бачити інформацію про роботу бота в консолі
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    # Запускаємо головну асинхронну функцію main()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Обробка зупинки бота через Ctrl+C
        print("Бота зупинено вручну.")
    except Exception as e:
        logging.error(f"Критична помилка: {e}", exc_info=True)
