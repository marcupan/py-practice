from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

from flask import Flask, abort, render_template


def _now_iso() -> str:
    """Return current UTC time in ISO-like format suitable for display.

    Using timezone-aware UTC time makes behavior predictable across environments.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class Config:
    """Base configuration for the Flask app.

    DEBUG can be toggled via the environment variable FLASK_DEBUG (0/1/true/false).
    """

    DEBUG: bool = os.getenv("FLASK_DEBUG", "1").lower() in {"1", "true", "yes"}


# --- Дані для нашого блогу/портфоліо (замість бази даних) ---
# Список словників, де кожен словник представляє пост. Робимо кортеж для
# імунтабельності на рівні модуля (мінімізуємо випадкові зміни глобального стану).
POSTS: Tuple[Dict[str, Any], ...] = (
    {
        "id": 1,
        "title": "Мій перший пост!",
        "content": "Це контент мого першого поста в блозі. Ласкаво просимо!",
        "author": "Адміністратор",
        "date": "2025-04-10",
    },
    {
        "id": 2,
        "title": "Знайомство з Flask",
        "content": "Flask - це чудовий мікрофреймворк для створення веб-додатків на Python. Він простий та гнучкий.",
        "author": "Розробник",
        "date": "2025-04-11",
    },
    {
        "id": 3,
        "title": "Проект Портфоліо",
        "content": "Тут може бути опис вашого проекту портфоліо, технології, які використовувались, та посилання.",
        "author": "Я",
        "date": "2025-04-12",
    },
)


def create_app(config: Optional[type[Config]] = None) -> Flask:
    """Application factory following Flask best practices.

    This allows other tools (tests, WSGI servers) to create the app
    without executing module-level side effects.
    """
    app = Flask(__name__)
    app.config.from_object(config or Config)

    # --- Маршрути (Routes) та функції-обробники (View Functions) ---

    @app.route("/")
    def index():
        """Обробник для головної сторінки. Показує список постів."""
        current_time = _now_iso()
        return render_template("index.html", posts=list(POSTS), current_time=current_time)

    @app.route("/about")
    def about():
        """Обробник для сторінки 'Про мене'."""
        current_time = _now_iso()
        return render_template("about.html", current_time=current_time)

    @app.route("/post/<int:post_id>")
    def show_post(post_id: int):
        """Обробник для відображення окремого поста."""
        post = next((p for p in POSTS if p["id"] == post_id), None)
        if post is None:
            abort(404)
        current_time = _now_iso()
        return render_template("post.html", post=post, current_time=current_time)

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):  # type: ignore[unused-argument]
        # Спробуємо показати дружню сторінку 404, якщо вона є; інакше - текст за замовчуванням
        try:
            return render_template("404.html"), 404
        except Exception:
            return "404 Not Found", 404

    return app


# --- Запуск додатку ---
# Цей блок виконується, лише якщо файл запускається напряму (python app.py)
if __name__ == "__main__":
    app = create_app()
    # ВАЖЛИВО: Не використовуйте DEBUG=True у production. Керуйте через FLASK_DEBUG.
    app.run(debug=bool(app.config.get("DEBUG", False)))
