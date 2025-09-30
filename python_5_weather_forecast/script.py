import requests  # Бібліотека для здійснення HTTP-запитів
import json  # Бібліотека для роботи з JSON
import datetime  # Для можливого перетворення часу
import os  # Для читання змінних середовища
from typing import Optional


def is_yes(answer: str) -> bool:
    """Повертає True, якщо відповідь користувача означає «так».
    Підтримує декілька поширених варіантів: "так", "y", "yes", "т".
    """
    if not isinstance(answer, str):
        return False
    return answer.strip().lower() in {"так", "y", "yes", "т"}


def get_api_key() -> Optional[str]:
    """Повертає API ключ з середовища або запиту користувача.
    Спочатку читає змінну середовища OPENWEATHER_API_KEY, інакше пропонує ввести вручну.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key:
        return api_key.strip()
    api_key = input("Введіть ваш API ключ OpenWeatherMap: ").strip()
    return api_key or None


# --- Функція для отримання та відображення погоди ---
def get_weather_forecast(api_key, city_name):
    """
    Отримує дані про погоду з OpenWeatherMap API та виводить їх.

    Args:
        api_key (str): Ваш унікальний ключ API від OpenWeatherMap.
        city_name (str): Назва міста, для якого потрібен прогноз.

    Returns:
        None: Функція друкує результат або повідомлення про помилку.
    """
    # Базова URL для поточного прогнозу погоди OpenWeatherMap API v2.5
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    # Параметри запиту:
    # q - назва міста
    # appid - ваш API ключ
    # units=metric - для отримання температури в градусах Цельсія
    # lang=uk - для отримання опису погоди українською мовою (якщо підтримується)
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric',
        'lang': 'uk'
    }

    try:
        # Виконуємо GET-запит до API
        response = requests.get(base_url, params=params, timeout=10)
        # Перевіряємо статус-код відповіді
        response.raise_for_status()  # Генерує помилку HTTPError для поганих статусів (4xx або 5xx)

        # Розбираємо JSON-відповідь у словник Python
        weather_data = response.json()

        # --- Витягуємо необхідні дані з відповіді ---
        # Використовуємо .get() для безпечного доступу, щоб уникнути KeyError
        main_data = weather_data.get('main', {})
        wind_data = weather_data.get('wind', {})
        weather_info = weather_data.get('weather', [{}])[0]  # Беремо перший елемент списку weather
        sys_data = weather_data.get('sys', {})

        city = weather_data.get('name', 'Невідоме місто')
        country = sys_data.get('country', '')
        description = weather_info.get('description', 'Опис недоступний').capitalize()
        temp = main_data.get('temp')
        feels_like = main_data.get('feels_like')
        humidity = main_data.get('humidity')
        pressure_hpa = main_data.get('pressure')
        wind_speed = wind_data.get('speed')  # м/с

        # Конвертуємо тиск з гПа в мм рт. ст. (приблизно 1 гПа = 0.750062 мм рт. ст.)
        pressure_mmhg = None
        if pressure_hpa is not None:
            pressure_mmhg = round(pressure_hpa * 0.750062)

        # --- Додаткові поля ---
        clouds = weather_data.get('clouds', {}).get('all')  # % хмарності
        timezone_shift = weather_data.get('timezone', 0)
        sunrise_ts = sys_data.get('sunrise')
        sunset_ts = sys_data.get('sunset')

        def fmt_time(ts):
            try:
                base = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                local_dt = base + datetime.timedelta(seconds=timezone_shift)
                return local_dt.strftime('%Y-%m-%d %H:%M')
            except Exception:
                return None

        sunrise_str = fmt_time(sunrise_ts) if sunrise_ts else None
        sunset_str = fmt_time(sunset_ts) if sunset_ts else None

        # --- Форматуємо та виводимо результат ---
        print("\n--- Погода ---")
        print(f"Місто: {city}, {country}")
        print(f"Опис: {description}")

        # Виводимо дані, лише якщо вони існують
        if temp is not None:
            print(f"Температура: {round(temp, 1)}°C")
        if feels_like is not None:
            print(f"Відчувається як: {round(feels_like, 1)}°C")
        if humidity is not None:
            print(f"Вологість: {humidity}%")
        if pressure_mmhg is not None:
            print(f"Тиск: {pressure_mmhg} мм рт. ст.")
        if wind_speed is not None:
            print(f"Швидкість вітру: {wind_speed} м/с")
        if clouds is not None:
            print(f"Хмарність: {clouds}%")
        if sunrise_str:
            print(f"Схід сонця: {sunrise_str}")
        if sunset_str:
            print(f"Захід сонця: {sunset_str}")
        print("-" * 14 + "\n")

    except requests.exceptions.HTTPError as http_err:
        # Обробка помилок HTTP (наприклад, 404 - місто не знайдено, 401 - невірний ключ)
        if response.status_code == 401:
            print("Помилка: Невірний API ключ. Будь ласка, перевірте ваш ключ.")
        elif response.status_code == 404:
            print(f"Помилка: Місто '{city_name}' не знайдено.")
        else:
            print(f"Помилка HTTP: {http_err}")  # Інші HTTP помилки
    except requests.exceptions.ConnectionError:
        print("Помилка: Не вдалося підключитися до сервера погоди. Перевірте з'єднання з Інтернетом.")
    except requests.exceptions.Timeout:
        print("Помилка: Час очікування відповіді від сервера вичерпано.")
    except requests.exceptions.RequestException as req_err:
        # Обробка інших помилок, пов'язаних з бібліотекою requests
        print(f"Помилка запиту: {req_err}")
    except json.JSONDecodeError:
        print("Помилка: Не вдалося розібрати відповідь від сервера.")
    except KeyError as key_err:
        # Ця помилка менш імовірна при використанні .get(), але залишаємо про всяк випадок
        print(f"Помилка: Не вдалося знайти ключ '{key_err}' у відповіді API.")
    except Exception as e:
        # Обробка будь-яких інших непередбачених помилок
        print(f"Сталася невідома помилка: {e}")


# --- Основна частина програми ---

def main():
    print("Програма прогнозу погоди")
    print("Для роботи потрібен API ключ від OpenWeatherMap (openweathermap.org)")
    print("Безкоштовний ключ можна отримати після реєстрації на сайті.")

    # Отримуємо API ключ (зі змінної середовища або запитуємо)
    api_key = get_api_key()

    if not api_key:
        print("API ключ не введено. Завершення програми.")
        return

    # Головний цикл програми
    while True:
        city = input("Введіть назву міста (або 'exit' для завершення): ").strip()
        if city.lower() == 'exit':
            break  # Вихід з циклу, якщо користувач ввів 'exit'
        if not city:
            print("Назва міста не може бути порожньою.")
            continue  # Пропускаємо ітерацію і запитуємо знову

        # Викликаємо функцію для отримання погоди
        get_weather_forecast(api_key, city)

    print("\nДякуємо за використання програми!")


if __name__ == "__main__":
    main()
