import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Модулі Scikit-learn
from sklearn.preprocessing import StandardScaler  # Для масштабування ознак
from sklearn.cluster import KMeans  # Алгоритм кластеризації
from sklearn.metrics import silhouette_score  # Метрика для оцінки якості кластеризації (опціонально)

# Для ігнорування попереджень
import warnings

warnings.filterwarnings("ignore")

print("Бібліотеки імпортовано.")

# --- 1. Завантаження даних ---
file_path = 'Mall_Customers.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Дані успішно завантажено з '{file_path}'. Розмір: {df.shape}")
except FileNotFoundError:
    print(f"Помилка: Файл '{file_path}' не знайдено.")
    print("Будь ласка, завантажте датасет 'Mall Customer Segmentation Data' з Kaggle.")
    exit()

# --- 2. Початкове дослідження даних (EDA) ---
print("\n--- 2. Початкове дослідження даних (EDA) ---")

print("\nПерші 5 рядків даних:")
print(df.head())

# Перейменуємо колонки для зручності
df.rename(columns={
    'Annual Income (k$)': 'AnnualIncome',
    'Spending Score (1-100)': 'SpendingScore'
}, inplace=True)

print("\nІнформація про DataFrame:")
df.info()
# Цей датасет зазвичай чистий і не має пропущених значень

print("\nОписові статистики:")
print(df.describe())

# Візуалізуємо зв'язки між основними ознаками
print("\nПобудова Pairplot для візуального аналізу зв'язків...")
sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender', palette='viridis', aspect=1.5)
plt.suptitle('Pairplot для аналізу даних клієнтів', y=1.02)
plt.show()
# Звертаємо увагу на графік 'AnnualIncome' vs 'SpendingScore' - там візуально видно потенційні кластери

# --- 3. Підготовка даних для Кластеризації ---
print("\n--- 3. Підготовка даних для Кластеризації ---")

# 3.1 Вибір ознак
# Для наочної демонстрації кластеризації візьмемо дві ключові ознаки:
# Річний дохід та Оцінка витрат.
X = df[['AnnualIncome', 'SpendingScore']].copy()
print("Обрано ознаки: 'AnnualIncome' та 'SpendingScore'.")

# 3.2 Масштабування ознак
# K-Means є алгоритмом, що базується на відстані, тому масштабування ознак
# є КРИТИЧНО важливим, щоб ознака з більшим діапазоном (AnnualIncome)
# не домінувала над ознакою з меншим діапазоном (SpendingScore).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Дані успішно масштабовано за допомогою StandardScaler.")

# --- 4. Визначення Оптимальної Кількості Кластерів (K) ---
print("\n--- 4. Визначення оптимальної кількості кластерів (K) ---")

# Використовуємо "Метод Ліктя" (Elbow Method)
# Рахуємо WCSS (Within-Cluster Sum of Squares) для різної кількості кластерів k.
# WCSS - це сума квадратів відстаней від кожної точки до центру її кластера.
wcss = []
k_range = range(1, 11)  # Перевіримо k від 1 до 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    # init='k-means++' - розумна ініціалізація центроїдів
    # random_state=42 - для відтворюваності результатів
    # n_init=10 - для уникнення локальних мінімумів
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ - це і є WCSS

# Будуємо графік методу ліктя
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Метод Ліктя для визначення оптимального K')
plt.xlabel('Кількість кластерів (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(k_range)
plt.grid(True)
plt.show()
# На графіку шукаємо "лікоть" - точку, де зменшення WCSS стає значно повільнішим.
# Для цього датасету "лікоть" чітко видно при K=5.
optimal_k = 5
print(f"\nЗа методом ліктя, оптимальна кількість кластерів: {optimal_k}")

# --- 5. Побудова та Тренування моделі K-Means ---
print(f"\n--- 5. Тренування моделі K-Means з K={optimal_k} ---")

# Створюємо та навчаємо модель з оптимальною кількістю кластерів
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
# fit_predict навчає модель і одразу повертає мітки кластерів для кожної точки
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Додаємо отримані мітки кластерів назад до вихідного DataFrame
df['Cluster'] = cluster_labels
print("Додано колонку 'Cluster' з мітками кластерів до DataFrame.")

print("\nПерші 5 рядків даних з мітками кластерів:")
print(df.head())

# --- 6. Візуалізація та Інтерпретація Кластерів ---
print("\n--- 6. Візуалізація та Інтерпретація Кластерів ---")

# Отримуємо координати центрів кластерів
# Їх потрібно повернути до вихідного масштабу для візуалізації
centroids_scaled = kmeans_final.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
print("\nКоординати центрів кластерів (у вихідному масштабі):")
print(pd.DataFrame(centroids, columns=['AnnualIncome', 'SpendingScore']))

# Будуємо діаграму розсіювання з кольоровими кластерами
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='viridis', s=100, alpha=0.8,
                legend='full')

# Наносимо центри кластерів на графік
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Центроїди')

plt.title('Сегментація Клієнтів за Доходом та Оцінкою Витрат')
plt.xlabel('Річний дохід (тис. $)')
plt.ylabel('Оцінка витрат (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# Інтерпретація результатів
print("\n--- Інтерпретація сегментів ---")
# Аналізуємо середні значення ознак для кожного кластера
cluster_analysis = df.drop('CustomerID', axis=1).groupby('Cluster')[
    ['Age', 'AnnualIncome', 'SpendingScore']].mean().round(2)
print(cluster_analysis)

print("\nМожлива інтерпретація кластерів:")
print("Кластер 0: 'Економні' - Середній дохід, низькі витрати.")
print("Кластер 1: 'Стандартні' - Середній дохід, середні витрати (ядро клієнтської бази).")
print("Кластер 2: 'Марнотрати' - Високий дохід, високі витрати (цільова/VIP аудиторія).")
print("Кластер 3: 'Обережні' - Низький дохід, низькі витрати.")
print("Кластер 4: 'Молоді транжири' - Низький дохід, високі витрати (можливо, молодь).")

print("\n--- Сегментацію клієнтів завершено ---")
