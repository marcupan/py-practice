import os
from pathlib import Path
import pandas as pd  # Бібліотека для роботи з даними (таблиці)
import matplotlib  # Основна бібліотека для візуалізації (бекенд налаштовується нижче)
matplotlib.use("Agg")  # Без GUI-бекенд для середовищ без екрана (CI/сервери)
import matplotlib.pyplot as plt
import seaborn as sns  # Бібліотека для покращеної візуалізації (надбудова над matplotlib)
import numpy as np  # Бібліотека для числових операцій (часто використовується pandas)

print("Бібліотеки успішно імпортовано.")

# --- Завантаження даних ---
# Seaborn має вбудовані набори даних для прикладів. Завантажимо 'tips'.
# Це DataFrame бібліотеки pandas.
print("Завантаження даних 'tips'...")
tips_df = None
try:
    tips_df = sns.load_dataset('tips')
    print("Дані успішно завантажено з seaborn.")
except Exception as e:
    print(f"Не вдалося завантажити з seaborn: {e}")
    local_csv = Path(__file__).parent / "tips.csv"
    if local_csv.exists():
        print(f"Спроба завантажити локальний файл: {local_csv}")
        tips_df = pd.read_csv(local_csv)
        print("Дані успішно завантажено з локального CSV.")
    else:
        print("Локальний файл не знайдено. Створюємо синтетичний набір даних.")
        rng = np.random.default_rng(42)
        size = 244  # приблизно як у tips
        tips_df = pd.DataFrame({
            'total_bill': rng.normal(19.8, 8.9, size).clip(3, 60),
            'tip': rng.normal(3.0, 1.4, size).clip(0.5, 12),
            'sex': rng.choice(['Male', 'Female'], size=size),
            'smoker': rng.choice(['Yes', 'No'], size=size, p=[0.2, 0.8]),
            'day': rng.choice(['Thur', 'Fri', 'Sat', 'Sun'], size=size, p=[0.2, 0.1, 0.4, 0.3]),
            'time': rng.choice(['Lunch', 'Dinner'], size=size, p=[0.3, 0.7]),
            'size': rng.integers(1, 6, size=size)
        })
        print("Синтетичні дані створено.")

# --- 1. Дослідження даних (Data Exploration) ---
print("\n--- 1. Дослідження даних ---")

# Подивимось на перші 5 рядків даних
print("\nПерші 5 рядків даних (head):")
print(tips_df.head())

# Отримаємо загальну інформацію про DataFrame (типи даних, кількість непустих значень)
print("\nЗагальна інформація про дані (info):")
tips_df.info()

# Отримаємо описові статистики для числових колонок
# (кількість, середнє, стандартне відхилення, мін/макс, квартилі)
print("\nОписові статистики (describe):")
print(tips_df.describe())

# Перевіримо наявність пропущених значень у кожній колонці
print("\nКількість пропущених значень по колонках (isnull().sum()):")
print(tips_df.isnull().sum())
# У цьому датасеті немає пропущених значень.

# Подивимось на унікальні значення в категоріальних колонках
print("\nУнікальні значення в колонці 'day':")
print(tips_df['day'].unique())
print("\nУнікальні значення в колонці 'time':")
print(tips_df['time'].unique())

# --- 2. Аналіз даних (Data Analysis) ---
print("\n--- 2. Аналіз даних ---")

# Розрахуємо середній розмір рахунку (total_bill) для кожного дня тижня
average_bill_per_day = tips_df.groupby('day', observed=False)['total_bill'].mean()
print("\nСередній рахунок по днях тижня:")
print(average_bill_per_day)

# Розрахуємо середній розмір чайових (tip) залежно від статі (sex) та факту паління (smoker)
average_tip_by_sex_smoker = tips_df.groupby(['sex', 'smoker'], observed=False)['tip'].mean().unstack()
# .unstack() робить таблицю зручнішою для читання
print("\nСередні чайові за статтю та фактом паління:")
print(average_tip_by_sex_smoker)

# Порахуємо кількість відвідувачів у різний час (Обід/Вечеря)
time_counts = tips_df['time'].value_counts()
print("\nКількість відвідувань за часом (Обід/Вечеря):")
print(time_counts)

# Розрахуємо відсоток чайових від загального рахунку
tips_df['tip_percentage'] = (tips_df['tip'] / tips_df['total_bill']) * 100
print("\nДодано колонку 'tip_percentage'. Перші 5 рядків:")
print(tips_df.head())

# --- 3. Візуалізація даних (Data Visualization) ---
print("\n--- 3. Візуалізація даних ---")
output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Графіки будуть збережені у: {output_dir}")

# Встановлюємо стиль графіків seaborn для кращого вигляду
sns.set_theme(style="whitegrid")

# 3.1 Гістограма: Розподіл загальних сум рахунків (total_bill)
plt.figure(figsize=(10, 6))  # Задаємо розмір вікна графіка
sns.histplot(data=tips_df, x='total_bill', kde=True, bins=20)  # kde=True додає лінію щільності розподілу
plt.title('Розподіл сум рахунків (Total Bill Distribution)')
plt.xlabel('Сума рахунку ($)')
plt.ylabel('Кількість')
plt.tight_layout()
plt.savefig(output_dir / "01_total_bill_hist.png", dpi=150)
plt.close()

# 3.2 Діаграма розсіювання (Scatter Plot): Зв'язок між сумою рахунку та чайовими
plt.figure(figsize=(10, 6))
# hue='smoker' розфарбовує точки залежно від того, чи палить клієнт
sns.scatterplot(data=tips_df, x='total_bill', y='tip', hue='smoker', size='size',
                # size='size' змінює розмір точки залежно від кількості людей
                alpha=0.7)  # alpha - прозорість точок
plt.title('Залежність чайових від суми рахунку')
plt.xlabel('Сума рахунку ($)')
plt.ylabel('Чайові ($)')
plt.tight_layout()
plt.savefig(output_dir / "02_scatter_total_bill_vs_tip.png", dpi=150)
plt.close()

# 3.3 Стовпчаста діаграма (Bar Plot): Середні чайові по днях тижня
plt.figure(figsize=(10, 6))
# Розрахуємо середнє значення для кожного дня перед побудовою
# estimator=np.mean - це значення за замовчуванням, але можна вказати іншу функцію (np.median, np.sum)
# errorbar=None вимикає показ "вусів" помилок для простоти
sns.barplot(data=tips_df, x='day', y='tip', estimator=np.mean, errorbar=None, color='C0',
            order=['Thur', 'Fri', 'Sat', 'Sun'])  # Задаємо порядок днів
plt.title('Середні чайові по днях тижня')
plt.xlabel('День тижня')
plt.ylabel('Середні чайові ($)')
plt.tight_layout()
plt.savefig(output_dir / "03_avg_tip_by_day.png", dpi=150)
plt.close()

# 3.4 Коробковий графік (Box Plot): Розподіл відсотка чайових за часом доби
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips_df, x='time', y='tip_percentage', color='C1')
plt.title('Розподіл відсотка чайових за часом доби (Обід/Вечеря)')
plt.xlabel('Час доби')
plt.ylabel('Відсоток чайових (%)')
plt.tight_layout()
plt.savefig(output_dir / "04_tip_percentage_by_time_box.png", dpi=150)
plt.close()

# 3.5 Теплова карта (Heatmap): Кореляція між числовими змінними
# Спочатку оберемо лише числові колонки для розрахунку кореляції
numeric_df = tips_df.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()  # Розраховуємо матрицю кореляцій

plt.figure(figsize=(10, 8))
# annot=True показує значення кореляції на карті
# cmap='coolwarm' задає колірну схему
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Теплова карта кореляцій числових змінних')
plt.tight_layout()
plt.savefig(output_dir / "05_correlation_heatmap.png", dpi=150)
plt.close()

print("\nАналіз та візуалізацію завершено.")
