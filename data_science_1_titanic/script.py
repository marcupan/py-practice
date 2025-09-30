import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Бібліотеки pandas, numpy, matplotlib, seaborn імпортовано.")

# --- 1. Завантаження даних ---
# Вкажіть правильний шлях до вашого файлу train.csv, якщо він не в тій же папці
file_path = 'train.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Дані успішно завантажено з '{file_path}'.")
except FileNotFoundError:
    print(f"Помилка: Файл '{file_path}' не знайдено.")
    print("Будь ласка, завантажте train.csv з Kaggle Titanic Competition і помістіть його в папку зі скриптом.")
    exit()  # Вихід зі скрипта, якщо файл не знайдено

# --- 2. Початкове дослідження даних ---
print("\n--- 2. Початкове дослідження даних ---")

# Виведемо перші 5 рядків
print("\nПерші 5 рядків даних:")
print(df.head())

# Виведемо назви колонок
print("\nНазви колонок:")
print(df.columns)

# Виведемо загальну інформацію (типи даних, кількість непустих значень)
print("\nІнформація про DataFrame (df.info()):")
df.info()

# Виведемо описові статистики для числових колонок
print("\nОписові статистики (df.describe()):")
print(df.describe())

# Перевіримо кількість пропущених значень
print("\nКількість пропущених значень (NaN) по колонках:")
print(df.isnull().sum())
# Бачимо пропуски в Age, Cabin, Embarked

# --- 3. Базове очищення даних ---
print("\n--- 3. Базове очищення даних ---")

# Обробка пропущених значень в 'Age'
# Заповнимо пропуски медіанним значенням віку
# Медіана менш чутлива до викидів, ніж середнє
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"\nПропущені значення в 'Age' заповнено медіаною ({median_age:.2f}).")

# Обробка пропущених значень в 'Embarked'
# Тут лише 2 пропуски. Заповнимо їх найчастішим значенням (модою).
most_frequent_embarked = df['Embarked'].mode()[0]  # mode() повертає Series, беремо перший елемент
df['Embarked'] = df['Embarked'].fillna(most_frequent_embarked)
print(f"Пропущені значення в 'Embarked' заповнено модою ('{most_frequent_embarked}').")

# Колонка 'Cabin' має занадто багато пропусків.
# Для базового аналізу ми можемо її поки ігнорувати або створити ознаку "має каюту/не має".
# Поки що просто відзначимо, що там багато пропусків.
print("Колонка 'Cabin' має багато пропусків і поки не обробляється.")

# Перевіримо пропуски ще раз
print("\nКількість пропущених значень після очищення:")
print(df.isnull().sum())

# --- 4. Дослідницький аналіз даних (EDA) та Візуалізація ---
print("\n--- 4. EDA та Візуалізація ---")
print("Зараз будуть відкриватися вікна з графіками...")

# Встановлюємо стиль для графіків
sns.set_theme(style="whitegrid")

# 4.1 Аналіз виживання (Survived) - цільова змінна
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Розподіл пасажирів за виживанням (0 = Загинув, 1 = Вижив)')
plt.xlabel('Статус виживання')
plt.ylabel('Кількість пасажирів')
survival_rate = df['Survived'].mean() * 100
print(f"\nЗагальний відсоток виживання: {survival_rate:.2f}%")
plt.tight_layout()
plt.show()

# 4.2 Виживання залежно від статі (Sex)
plt.figure(figsize=(7, 5))
sns.countplot(x='Sex', hue='Survived', data=df, palette='pastel')
plt.title('Виживання залежно від статі')
plt.xlabel('Стать')
plt.ylabel('Кількість пасажирів')
plt.legend(title='Статус', labels=['Загинув', 'Вижив'])
# Розрахуємо відсоток виживання для кожної статі
survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
print("\nВідсоток виживання за статтю:")
print(survival_by_sex)
plt.tight_layout()
plt.show()

# 4.3 Виживання залежно від класу каюти (Pclass)
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='deep')
plt.title('Виживання залежно від класу каюти')
plt.xlabel('Клас каюти (1 = Вищий, 3 = Нижчий)')
plt.ylabel('Кількість пасажирів')
plt.legend(title='Статус', labels=['Загинув', 'Вижив'])
# Розрахуємо відсоток виживання для кожного класу
survival_by_pclass = df.groupby('Pclass')['Survived'].mean() * 100
print("\nВідсоток виживання за класом каюти:")
print(survival_by_pclass)
plt.tight_layout()
plt.show()

# 4.4 Розподіл віку пасажирів (Age)
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30, color='skyblue')  # kde=True додає лінію щільності
plt.title('Розподіл віку пасажирів (після заповнення NaN)')
plt.xlabel('Вік (роки)')
plt.ylabel('Кількість пасажирів')
plt.tight_layout()
plt.show()

# 4.5 Розподіл віку залежно від статусу виживання
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30, palette='muted')
plt.title('Розподіл віку для тих, хто вижив та загинув')
plt.xlabel('Вік (роки)')
plt.ylabel('Кількість пасажирів')
plt.legend(title='Статус', labels=['Вижив', 'Загинув'])  # Порядок може бути іншим залежно від версії seaborn
plt.tight_layout()
plt.show()

# 4.6 Розподіл вартості квитка (Fare)
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'], kde=True, bins=40, color='lightcoral')
# Через великі викиди логарифмування може бути корисним для візуалізації, але поки покажемо як є
plt.title('Розподіл вартості квитка (Fare)')
plt.xlabel('Вартість квитка')
plt.ylabel('Кількість пасажирів')
# plt.xlim(0, 300) # Можна обмежити вісь X для кращої видимості основної маси
plt.tight_layout()
plt.show()

# 4.7 Виживання залежно від порту посадки (Embarked)
plt.figure(figsize=(8, 5))
sns.countplot(x='Embarked', hue='Survived', data=df, palette='bright', order=['S', 'C', 'Q'])  # Явно задаємо порядок
plt.title('Виживання залежно від порту посадки')
plt.xlabel('Порт посадки (S=Southampton, C=Cherbourg, Q=Queenstown)')
plt.ylabel('Кількість пасажирів')
plt.legend(title='Статус', labels=['Загинув', 'Вижив'])
survival_by_embarked = df.groupby('Embarked')['Survived'].mean() * 100
print("\nВідсоток виживання за портом посадки:")
print(survival_by_embarked)
plt.tight_layout()
plt.show()

# 4.8 Взаємозв'язок між класом каюти та віком
plt.figure(figsize=(9, 6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Розподіл віку пасажирів за класом каюти')
plt.xlabel('Клас каюти')
plt.ylabel('Вік (роки)')
plt.tight_layout()
plt.show()

print("\n--- Аналіз та візуалізація даних 'Титаніка' завершені ---")
