from flask import Flask, render_template, abort
import datetime  # Імпортуємо модуль datetime для відображення поточного часу

# Створюємо екземпляр додатку Flask
# __name__ допомагає Flask знайти шаблони та статичні файли
app = Flask(__name__)

# --- Дані для нашого блогу/портфоліо (замість бази даних) ---
# Список словників, де кожен словник представляє пост
POSTS = [
    {
        'id': 1,
        'title': 'Мій перший пост!',
        'content': 'Це тіло мого першого поста в блозі. Ласкаво просимо!',
        'author': 'Адміністратор',
        'date': '2025-04-10'
    },
    {
        'id': 2,
        'title': 'Знайомство з Flask',
        'content': 'Flask - це чудовий мікрофреймворк для створення веб-додатків на Python. Він простий та гнучкий.',
        'author': 'Розробник',
        'date': '2025-04-11'
    },
    {
        'id': 3,
        'title': 'Проект Портфоліо',
        'content': 'Тут може бути опис вашого проекту портфоліо, технології, які використовувались, та посилання.',
        'author': 'Я',
        'date': '2025-04-12'
    }
]


# --- Кінець даних ---

# --- Маршрути (Routes) та функції-обробники (View Functions) ---

# Маршрут для головної сторінки ('/')
@app.route('/')
def index():
    """Обробник для головної сторінки. Показує список постів."""
    # Передаємо список POSTS у шаблон index.html
    # Також додамо поточний час для демонстрації в base.html
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('index.html', posts=POSTS, current_time=current_time)


# Маршрут для сторінки "Про мене" ('/about')
@app.route('/about')
def about():
    """Обробник для сторінки 'Про мене'."""
    # Просто відображаємо статичний шаблон about.html
    # Також передаємо поточний час
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('about.html', current_time=current_time)


# Маршрут для перегляду окремого поста за його ID ('/post/<id>')
# <int:post_id> - це змінна частина URL, яка очікує ціле число (id поста)
@app.route('/post/<int:post_id>')
def show_post(post_id):
    """Обробник для відображення окремого поста."""
    # Шукаємо пост з відповідним ID у нашому списку POSTS
    # next() з None для уникнення помилки, якщо пост не знайдено
    post = next((p for p in POSTS if p['id'] == post_id), None)

    if post is None:
        # Якщо пост з таким ID не знайдено, повертаємо помилку 404 Not Found
        abort(404)

    # Якщо пост знайдено, передаємо його у шаблон post.html
    # Також передаємо поточний час
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('post.html', post=post, current_time=current_time)


# --- Запуск додатку ---
# Цей блок виконується, лише якщо файл запускається напряму (python app.py)
if __name__ == '__main__':
    # app.run() запускає локальний сервер для розробки
    # debug=True вмикає режим налагодження:
    # - Автоматичне перезавантаження сервера при зміні коду
    # - Детальні повідомлення про помилки в браузері
    # ВАЖЛИВО: Ніколи не використовуйте debug=True в робочому (production) середовищі!
    app.run(debug=True)
