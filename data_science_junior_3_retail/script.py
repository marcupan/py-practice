import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt  # Для роботи з датами та часом

print("Бібліотеки pandas, numpy, matplotlib, seaborn, datetime імпортовано.")

# --- 1. Завантаження даних ---
# Вкажіть правильний шлях та назву вашого файлу
file_path = 'online-retail.xlsx'

try:
    # Спробуємо прочитати Excel файл. Якщо у вас CSV, використовуйте pd.read_csv
    # Важливо вказати кодування, оскільки датасет часто містить не-ASCII символи
    # Поширені кодування для цього датасету: 'ISO-8859-1', 'latin1', 'cp1252'
    try:
        df = pd.read_excel(file_path)
    except ValueError as e:  # Спроба з CSV, якщо Excel не спрацював або помилка формату
        if 'Excel file format cannot be determined' in str(e) or isinstance(e, FileNotFoundError):
            print(f"Не вдалося прочитати як Excel. Спроба прочитати як CSV: {file_path.replace('.xlsx', '.csv')}")
            file_path = file_path.replace('.xlsx', '.csv')  # Спробуємо ім'я CSV
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        else:
            raise e  # Якщо інша помилка Excel, піднімаємо її

    print(f"Дані успішно завантажено з '{file_path}'.")
    print(f"Розмірність даних: {df.shape[0]} рядків, {df.shape[1]} колонок")

except FileNotFoundError:
    print(f"Помилка: Файл '{file_path}' не знайдено.")
    print(
        "Будь ласка, завантажте датасет 'Online Retail' (UCI ML Repository або Kaggle) і помістіть його в папку зі скриптом.")
    exit()
except Exception as e:
    print(f"Сталася помилка при читанні файлу: {e}")
    exit()

# --- 2. Початкове дослідження даних ---
print("\n--- 2. Початкове дослідження даних ---")

print("\nПерші 5 рядків даних:")
print(df.head())

print("\nІнформація про DataFrame (df.info()):")
df.info()
# Звертаємо увагу на типи даних (InvoiceDate має бути datetime, CustomerID - float через NaN)

print("\nОписові статистики (df.describe()):")
print(df.describe())
# Бачимо від'ємну Quantity (повернення) та нульову UnitPrice

print("\nКількість пропущених значень (NaN) по колонках:")
print(df.isnull().sum())
# Значна кількість пропусків у CustomerID, трохи в Description

# --- 3. Очищення та Передобробка даних ---
print("\n--- 3. Очищення та Передобробка даних ---")

# 3.1 Перетворення типів та видалення некоректних даних
# Перетворюємо 'InvoiceDate' на datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print("\nКолонку 'InvoiceDate' перетворено на тип datetime.")

# Видаляємо рядки з відсутнім CustomerID для аналізу за клієнтами (але збережемо оригінал для загальних продажів)
# df_cleaned = df.dropna(subset=['CustomerID'])
# print(f"Видалено {df.shape[0] - df_cleaned.shape[0]} рядків з відсутнім CustomerID.")
# Поки що залишимо всі рядки для аналізу загальних продажів, але пам'ятаємо про NaN в CustomerID

# Видаляємо транзакції з від'ємною кількістю (повернення/скасування)
df_original_rows = df.shape[0]
df = df[df['Quantity'] > 0]
print(f"Видалено {df_original_rows - df.shape[0]} рядків з Quantity <= 0.")

# Видаляємо транзакції з нульовою ціною
df_original_rows = df.shape[0]
df = df[df['UnitPrice'] > 0]
print(f"Видалено {df_original_rows - df.shape[0]} рядків з UnitPrice <= 0.")

# Перевірка і видалення можливих дублікатів
df_original_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Видалено {df_original_rows - df.shape[0]} дублікатів рядків.")

# Перевіряємо пропуски після початкової чистки
print("\nКількість пропущених значень після початкової чистки:")
print(df.isnull().sum())  # CustomerID все ще має пропуски, якщо не видаляли

# 3.2 Інженерія ознак (Feature Engineering)
# Створюємо колонку загальної вартості (TotalPrice)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Витягуємо часові компоненти
df['InvoiceYearMonth'] = df['InvoiceDate'].dt.to_period('M')  # Рік-Місяць для агрегації
df['InvoiceMonth'] = df['InvoiceDate'].dt.month_name()  # Назва місяця
df['InvoiceDayOfWeek'] = df['InvoiceDate'].dt.day_name()  # Назва дня тижня
df['InvoiceHour'] = df['InvoiceDate'].dt.hour  # Година

print("Створено колонки 'TotalPrice', 'InvoiceYearMonth', 'InvoiceMonth', 'InvoiceDayOfWeek', 'InvoiceHour'.")
print("\nОновлені дані (перші 2 рядки):")
print(df[['InvoiceDate', 'TotalPrice', 'InvoiceYearMonth', 'InvoiceHour']].head(2))

# --- 4. EDA та Візуалізація ---
print("\n--- 4. EDA та Візуалізація ---")
print("Зараз будуть відкриватися вікна з графіками...")

sns.set_theme(style="darkgrid")

# 4.1 Динаміка загальних продажів за місяцями
monthly_revenue = df.groupby('InvoiceYearMonth')['TotalPrice'].sum()
# Перетворюємо PeriodIndex на рядки для графіка (або можна використовувати .timestamp())
monthly_revenue.index = monthly_revenue.index.to_timestamp()

plt.figure(figsize=(12, 6))
monthly_revenue.plot(kind='line', marker='o')
plt.title('Динаміка загальних продажів за місяцями')
plt.xlabel('Місяць')
plt.ylabel('Загальний дохід')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 4.2 Загальні продажі за днями тижня
plt.figure(figsize=(10, 6))
# Визначимо правильний порядок днів
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.barplot(data=df, x='InvoiceDayOfWeek', y='TotalPrice', estimator=sum, errorbar=None,
            order=day_order, palette='Blues_d')
plt.title('Загальний дохід за днями тижня')
plt.xlabel('День тижня')
plt.ylabel('Загальний дохід')
plt.tight_layout()
plt.show()

# 4.3 Загальні продажі за годинами дня
plt.figure(figsize=(12, 6))
hourly_revenue = df.groupby('InvoiceHour')['TotalPrice'].sum()
hourly_revenue.plot(kind='bar', color='skyblue')
plt.title('Загальний дохід за годинами дня')
plt.xlabel('Година дня (0-23)')
plt.ylabel('Загальний дохід')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4.4 Топ-10 товарів за кількістю продажів
top_products_quantity = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 7))
top_products_quantity.sort_values().plot(kind='barh', color='lightgreen')  # barh - горизонтальна
plt.title('Топ-10 товарів за кількістю проданих одиниць')
plt.xlabel('Загальна кількість проданих одиниць')
plt.ylabel('Опис товару')
plt.tight_layout()
plt.show()

# 4.5 Топ-10 товарів за загальним доходом
top_products_revenue = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 7))
top_products_revenue.sort_values().plot(kind='barh', color='coral')
plt.title('Топ-10 товарів за загальним доходом')
plt.xlabel('Загальний дохід')
plt.ylabel('Опис товару')
plt.tight_layout()
plt.show()

# 4.6 Топ-10 країн за загальним доходом (без UK для кращої видимості інших)
top_countries = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
# Виключимо UK, оскільки вона домінує
top_countries_no_uk = top_countries.drop('United Kingdom').head(10)

plt.figure(figsize=(12, 7))
top_countries_no_uk.sort_values().plot(kind='barh', color='gold')
plt.title('Топ-10 країн за загальним доходом (без Великої Британії)')
plt.xlabel('Загальний дохід')
plt.ylabel('Країна')
plt.tight_layout()
plt.show()

print("\nЗагальний дохід по країнах (включаючи UK):")
print(top_countries.head())

# 4.7 Розподіл вартості замовлень (на одну інвойс)
invoice_value = df.groupby('InvoiceNo')['TotalPrice'].sum()
plt.figure(figsize=(10, 6))
sns.histplot(invoice_value, bins=200, color='magenta')
plt.title('Розподіл вартості одного замовлення (інвойсу)')
plt.xlabel('Вартість замовлення')
plt.ylabel('Кількість замовлень')
plt.xlim(0, 1000)  # Обмежимо вісь X, щоб побачити основний розподіл
plt.tight_layout()
plt.show()
print(f"\nСередня вартість одного замовлення: {invoice_value.mean():.2f}")
print(f"Медіанна вартість одного замовлення: {invoice_value.median():.2f}")

print("\n--- Аналіз та візуалізація даних про продажі завершені ---")
