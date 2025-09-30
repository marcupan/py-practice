import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json  # Для роботи з JSON-даними в колонках
import ast  # Для безпечного перетворення рядка JSON у словник/список
from pathlib import Path
import os

print("Бібліотеки pandas, numpy, matplotlib, seaborn, json, ast імпортовано.")

# --- 1. Завантаження та об'єднання даних ---
BASE_DIR = Path(__file__).resolve().parent
movies_file_path = BASE_DIR / 'tmdb_5000_movies.csv'
credits_file_path = BASE_DIR / 'tmdb_5000_credits.csv'
outputs_dir = BASE_DIR / 'outputs'
outputs_dir.mkdir(exist_ok=True)

try:
    movies_df = pd.read_csv(movies_file_path)
    credits_df = pd.read_csv(credits_file_path)
    print(f"Дані успішно завантажено з '{movies_file_path.name}' та '{credits_file_path.name}'.")
except FileNotFoundError:
    print("Помилка: Один або обидва файли не знайдено.")
    print("Будь ласка, завантажте tmdb_5000_movies.csv та tmdb_5000_credits.csv з Kaggle і помістіть їх в папку зі скриптом.")
    raise

# Перейменуємо колонку для об'єднання в credits_df для ясності
credits_df.rename(columns={'movie_id': 'id'}, inplace=True)

# Об'єднаємо два датафрейми за колонкою 'id'
df = movies_df.merge(credits_df, on='id', suffixes=('_movies', '_credits'))
print(f"Датафрейми об'єднано. Загальна кількість фільмів: {df.shape[0]}, колонок: {df.shape[1]}")

# Нормалізуємо назву фільму до однієї колонки 'title'
if 'title_movies' in df.columns:
    df['title'] = df['title_movies']
elif 'title' in df.columns:
    df['title'] = df['title']
else:
    df['title'] = np.nan

# --- 2. Початкове дослідження об'єднаних даних ---
print("\n--- 2. Початкове дослідження даних ---")

print("\nПерші 2 рядки об'єднаних даних:")
try:
    from IPython.display import display

    display(df.head(2))
except ImportError:
    print(df.head(2))

print("\nНазви колонок:")
print(df.columns)

print("\nІнформація про DataFrame (df.info()):")
df.info()

print("\nОписові статистики (df.describe()):")
print(df.describe())

print("\nКількість пропущених значень (NaN) по колонках (топ 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# --- 3. Очищення та Передобробка даних ---
print("\n--- 3. Очищення та Передобробка даних ---")

# 3.1 Обробка пропущених значень (приклади)
median_runtime = df['runtime'].median()
df['runtime'] = df['runtime'].fillna(median_runtime)
print(f"\nПропущені значення 'runtime' заповнено медіаною ({median_runtime}).")

df.dropna(subset=['release_date'], inplace=True)
print("Видалено рядки з відсутньою 'release_date'.")

# 3.2 Перетворення типів даних
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
print("Створено колонки 'release_year' та 'release_month'.")

df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')


# 3.3 Робота з JSON-колонками (на прикладі 'genres')
def parse_json_list(data_str):
    try:
        data_list = ast.literal_eval(data_str)
        return [item['name'] for item in data_list]
    except (ValueError, SyntaxError, TypeError):
        return []


df['genres_list'] = df['genres'].apply(parse_json_list)
print("Розпарсено колонку 'genres' у список назв жанрів ('genres_list').")
print("Приклад розпарсених жанрів:")
print(df[['title', 'genres_list']].head(3))

# --- 4. EDA та Візуалізація ---
print("\n--- 4. EDA та Візуалізація ---")
print(f"Графіки будуть збережені в папку: {outputs_dir}")

sns.set_theme(style="whitegrid")

# 4.1 Найпопулярніші жанри
all_genres = df['genres_list'].explode().dropna()
plt.figure(figsize=(12, 8))
# Щоб уникнути FutureWarning у seaborn (palette без hue), використовуємо hue та вимикаємо легенду
genres_df = pd.DataFrame({'genre': all_genres})
order = genres_df['genre'].value_counts().index
ax = sns.countplot(data=genres_df, y='genre', order=order, hue='genre', palette='mako', legend=False)
# Для сумісності з версіями seaborn без параметра legend — видаляємо легенду вручну, якщо вона є
if hasattr(ax, 'legend_') and ax.legend_:
    ax.legend_.remove()
plt.title('Найпопулярніші жанри фільмів')
plt.xlabel('Кількість фільмів')
plt.ylabel('Жанр')
plt.tight_layout()
plt.savefig(outputs_dir / '01_top_genres.png', dpi=150)
plt.close()
print(f"Збережено графік: {outputs_dir / '01_top_genres.png'}")

# 4.2 Розподіл рейтингів фільмів (vote_average)
plt.figure(figsize=(10, 6))
sns.histplot(df['vote_average'], kde=False, bins=20, color='teal')
plt.title('Розподіл середніх рейтингів фільмів (TMDB)')
plt.xlabel('Середній рейтинг (0-10)')
plt.ylabel('Кількість фільмів')
plt.tight_layout()
plt.savefig(outputs_dir / '02_vote_average_hist.png', dpi=150)
plt.close()
print(f"Збережено графік: {outputs_dir / '02_vote_average_hist.png'}")

# 4.3 Розподіл тривалості фільмів (runtime)
plt.figure(figsize=(10, 6))
sns.histplot(df[df['runtime'] > 0]['runtime'], kde=True, bins=30, color='purple')
plt.title('Розподіл тривалості фільмів (хвилини)')
plt.xlabel('Тривалість (хв)')
plt.ylabel('Кількість фільмів')
plt.tight_layout()
plt.savefig(outputs_dir / '03_runtime_hist.png', dpi=150)
plt.close()
print(f"Збережено графік: {outputs_dir / '03_runtime_hist.png'}")

# 4.4 Кількість фільмів за роком виходу
plt.figure(figsize=(14, 7))
movies_per_year = df['release_year'].value_counts().sort_index()
movies_per_year = movies_per_year[movies_per_year.index > 1960]
movies_per_year.plot(kind='line', marker='o', linestyle='-')
plt.title('Кількість випущених фільмів за роком (після 1960)')
plt.xlabel('Рік виходу')
plt.ylabel('Кількість фільмів')
plt.grid(True)
plt.tight_layout()
plt.savefig(outputs_dir / '04_movies_per_year.png', dpi=150)
plt.close()
print(f"Збережено графік: {outputs_dir / '04_movies_per_year.png'}")

# 4.5 Залежність доходу від бюджету (Revenue vs Budget)
budget_revenue_df = df[(df['budget'] > 1000) & (df['revenue'] > 1000)]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=budget_revenue_df, x='budget', y='revenue', alpha=0.5)
plt.title('Залежність доходу від бюджету фільму (виключені нулі)')
plt.xlabel('Бюджет ($)')
plt.ylabel('Дохід ($)')
plt.tight_layout()
plt.savefig(outputs_dir / '05_revenue_vs_budget.png', dpi=150)
plt.close()
print(f"Збережено графік: {outputs_dir / '05_revenue_vs_budget.png'}")

# 4.6 Залежність рейтингу від бюджету
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df[df['budget'] > 1000], x='budget', y='vote_average', alpha=0.3)
plt.title('Залежність середнього рейтингу від бюджету')
plt.xlabel('Бюджет ($)')
plt.ylabel('Середній рейтинг (0-10)')
plt.tight_layout()
plt.savefig(outputs_dir / '06_rating_vs_budget.png', dpi=150)
plt.close()
print(f"Збережено графік: {outputs_dir / '06_rating_vs_budget.png'}")

# 4.7 Топ 10 найприбутковіших фільмів
df['profit'] = df['revenue'] - df['budget']
top_profit_movies = df.sort_values('profit', ascending=False).head(10)
print("\n--- Топ 10 найприбутковіших фільмів ---")
print(top_profit_movies[['title', 'profit', 'budget', 'revenue']])

# 4.8 Топ 10 фільмів з найвищим рейтингом (з урахуванням кількості голосів)
C = df['vote_average'].mean()
m = df['vote_count'].quantile(0.90)
print(f"\nМінімальна кількість голосів для топ-рейтингу (m): {int(m)}")
qualified_movies = df.copy().loc[df['vote_count'] >= m]


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m)) * R + (m / (v + m)) * C


qualified_movies['score'] = qualified_movies.apply(weighted_rating, axis=1)
qualified_movies = qualified_movies.sort_values('score', ascending=False)

print("\n--- Топ 10 фільмів за зваженим рейтингом (враховуючи кількість голосів) ---")
print(qualified_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))

print("\n--- Аналіз та візуалізація даних про фільми завершені ---")
