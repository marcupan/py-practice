import requests  # Бібліотека для здійснення HTTP-запитів
from bs4 import BeautifulSoup  # Бібліотека для парсингу HTML та XML документів
import time  # Для можливої затримки між запитами
from urllib.parse import urljoin  # Для обробки відносних URL
from typing import List, Dict, Optional


# --- Допоміжна функція для запиту з простими ретраями ---
def _get_with_retries(url: str, headers: dict, timeout: int = 10, retries: int = 2, backoff: float = 1.0) -> Optional[requests.Response]:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                break
        except requests.exceptions.HTTPError:
            # Для 4xx/5xx не повторюємо, просто кидаємо далі
            raise
    if last_exc:
        raise last_exc
    return None


# --- Функція для скрапінгу новин ---
def scrape_pravda_news(url: str = "https://www.pravda.com.ua/news/", limit: int = 20) -> List[Dict[str, str]]:
    """
    Скрапить заголовки та посилання на новини з вказаної URL (за замовчуванням - стрічка новин УП).

    Args:
        url (str): URL сторінки новин для скрапінгу.
        limit (int): Максимальна кількість новин для повернення.

    Returns:
        list: Список словників, де кожен словник містить 'title' та 'link' новини,
              або порожній список у разі помилки.
    """
    print(f"Спроба отримати новини з: {url}")
    news_list: List[Dict[str, str]] = []

    # Встановлюємо заголовок User-Agent, щоб імітувати браузер
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36'
    }

    try:
        # Виконуємо GET-запит до вказаної URL з заголовками з простими ретраями
        response = _get_with_retries(url, headers=headers, timeout=10, retries=2, backoff=1.0)

        # Встановлюємо кодування відповіді вручну на utf-8 (корисно для кирилиці)
        if response is None:
            return []
        response.encoding = 'utf-8'

        # Створюємо об'єкт BeautifulSoup для парсингу HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- Пошук елементів новин ---
        # 1) Шукаємо відомі контейнери; якщо не знайдено, працюємо по всій сторінці
        news_container = (
            soup.find('div', class_='article_news_list') or
            soup.find('div', class_='article_list') or
            soup.find('div', class_='section') or
            soup.find('main') or
            soup.find('div', id='container')
        )

        # 2) Витягуємо потенційні блоки/елементи
        if news_container:
            candidates = (
                news_container.find_all('div', class_='article_item') or
                news_container.find_all('article') or
                news_container.find_all('li')
            )
        else:
            # Якщо контейнер не знайдено — беремо елементи з усього документа
            candidates = soup.find_all(['article', 'li', 'div'])

        # 3) Як крайній випадок — напряму шукаємо посилання, що ведуть у /news/
        if not candidates:
            link_candidates = soup.find_all('a')
        else:
            link_candidates = []

        if not candidates and not link_candidates:
            print("Не знайдено елементів новин у контейнері за очікуваними селекторами.")
            return []

        print(f"Знайдено {len(candidates) if candidates else 0} потенційних блоків та {len(link_candidates) if link_candidates else 0} прямих посилань.")

        seen_links = set()
        def consider(title: str, href: str):
            if not (title and href):
                return False
            absolute = urljoin(url, href)
            # Відсіюємо сміття і дублікати; залишаємо лише посилання на pravda.com.ua і шлях /news/
            if 'pravda.com.ua' not in absolute:
                return False
            if '/news/' not in absolute:
                return False
            if absolute in seen_links:
                return False
            seen_links.add(absolute)
            news_list.append({'title': title, 'link': absolute})
            return True

        # Спершу проходимо по блоках і шукаємо всередині якірці з корисними класами
        for item in candidates or []:
            link_tag = (
                item.find('a', class_='article_link') or
                item.find('a', class_='news_all__link') or
                item.find('a', class_='news_item__title') or
                item.find('a', class_='list__title') or
                item.find('a')
            )
            if not link_tag:
                continue
            title = link_tag.get_text(strip=True)
            href = link_tag.get('href')
            if consider(title, href) and len(news_list) >= max(1, limit):
                break

        # Якщо ще не назбирали ліміт — пройдемося напряму по всіх <a>
        if len(news_list) < max(1, limit):
            anchors = link_candidates or soup.find_all('a')
            for a in anchors:
                title = a.get_text(strip=True)
                href = a.get('href')
                if consider(title, href) and len(news_list) >= max(1, limit):
                    break

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


def main():
    # Простий CLI для запуску
    default_url = "https://www.pravda.com.ua/news/"
    try:
        user_url = input(f"Введіть URL стрічки новин (Enter для за замовчуванням: {default_url}): ").strip()
    except EOFError:
        user_url = ''
    url = user_url or default_url

    try:
        limit_str = input("Скільки новин отримати? (за замовчуванням 20): ").strip()
        limit = int(limit_str) if limit_str else 20
    except Exception:
        limit = 20

    news = scrape_pravda_news(url=url, limit=limit)
    if news:
        print("\n--- Останні новини ---")
        for i, item in enumerate(news, start=1):
            print(f"{i}. Заголовок: {item['title']}")
            print(f"   Посилання: {item['link']}\n")
    else:
        print("Не вдалося отримати новини.")


# --- Приклад використання ---
if __name__ == "__main__":
    main()
