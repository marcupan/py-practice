import requests  # Бібліотека для здійснення HTTP-запитів
from bs4 import BeautifulSoup  # Бібліотека для парсингу HTML та XML документів
import time  # Для можливої затримки між запитами (хороша практика)
from urllib.parse import urljoin  # Для обробки відносних URL


# --- Функція для скрапінгу новин ---
def scrape_pravda_news(url="https://www.pravda.com.ua/news/"):
    """
    Скрапить заголовки та посилання на новини з вказаної URL (за замовчуванням - стрічка новин УП).

    Args:
        url (str): URL сторінки новин для скрапінгу.

    Returns:
        list: Список словників, де кожен словник містить 'title' та 'link' новини,
              або порожній список у разі помилки.
    """
    print(f"Спроба отримати новини з: {url}")
    news_list = []

    # Встановлюємо заголовок User-Agent, щоб імітувати браузер
    # Деякі сайти можуть блокувати запити без цього заголовка
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Затримка перед запитом (ввічливість до сервера)
        # time.sleep(1) # Розкоментуйте, якщо плануєте робити багато запитів

        # Виконуємо GET-запит до вказаної URL з заголовками
        response = requests.get(url, headers=headers, timeout=10)  # timeout - час очікування відповіді

        # Перевіряємо, чи запит був успішним (статус код 200 OK)
        # Якщо ні, генерується виняток HTTPError
        response.raise_for_status()

        # Встановлюємо кодування відповіді вручну на utf-8, якщо автоматичне визначення некоректне
        # Це часто допомагає з кирилицею на деяких сайтах
        response.encoding = 'utf-8'

        # Створюємо об'єкт BeautifulSoup для парсингу HTML
        # 'html.parser' - стандартний парсер Python
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- Пошук елементів новин ---
        # На основі актуальної структури pravda.com.ua/news/:
        # Новини знаходяться в div з класом 'article_news_list' або 'article_list'.
        # Кожен окремий елемент новини має клас 'article_item'.
        # Заголовок новини є посиланням (тег 'a') з класом 'article_link' всередині 'article_item'.

        # Спробуємо знайти контейнер списку новин
        news_container = soup.find('div', class_='article_news_list')
        if not news_container:
            news_container = soup.find('div', class_='article_list')

        if not news_container:
            print(
                "Не знайдено основного контейнера новин ('article_news_list' або 'article_list'). Можливо, структура сайту змінилась.")
            return []

        # Тепер шукаємо окремі елементи новин всередині контейнера
        news_items = news_container.find_all('div', class_='article_item')

        if not news_items:
            print(f"Не знайдено елементів новин за селектором 'div' з класом 'article_item' у контейнері.")
            # Якщо article_item не спрацював, спробуємо знайти посилання напряму,
            # хоча це менш точно, бо може захопити зайве.
            # Наприклад, якщо посилання мають спільний клас на інших сторінках УП.
            # news_items = soup.find_all('a', class_='article_link')
            return []

        print(f"Знайдено {len(news_items)} потенційних блоків/посилань новин.")

        # Проходимо по знайдених елементах новин
        for item in news_items:
            # Шукаємо тег 'a' з класом 'article_link' всередині кожного елемента новини
            link_tag = item.find('a', class_='article_link')

            if link_tag:
                # Отримуємо текст заголовка, видаляючи зайві пробіли по краях
                title = link_tag.get_text(strip=True)
                # Отримуємо значення атрибута 'href' (посилання)
                link = link_tag.get('href')

                # Перевіряємо, чи отримали і заголовок, і посилання
                if title and link:
                    # Перетворюємо відносні URL на абсолютні
                    # urljoin об'єднує базову URL сайту з відносним посиланням
                    absolute_link = urljoin(url, link)

                    # Додаємо словник з новиною до нашого списку
                    news_list.append({'title': title, 'link': absolute_link})
            else:
                # Цей блок виконається, якщо всередині 'article_item' не знайдено посилання з 'article_link'
                # Може бути корисним для налагодження, якщо зміниться структура
                # print(f"Попередження: Не знайдено посилання з класом 'article_link' в елементі: {item}")
                pass

        # Обмежимо кількість новин для виводу, якщо їх забагато
        news_list = news_list[:20]  # Показуємо перші 20 знайдених новин

    except requests.exceptions.HTTPError as errh:
        print(f"Помилка HTTP: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Помилка підключення: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Таймаут запиту: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Сталася непередбачена помилка запиту: {err}")
    except Exception as e:
        print(f"Виникла невідома помилка: {e}")

    return news_list


# --- Приклад використання ---
if __name__ == "__main__":
    pravda_news = scrape_pravda_news()

    if pravda_news:
        print("\n--- Останні новини з pravda.com.ua ---")
        for i, news in enumerate(pravda_news):
            print(f"{i + 1}. Заголовок: {news['title']}")
            print(f"   Посилання: {news['link']}\n")
    else:
        print("Не вдалося отримати новини.")
