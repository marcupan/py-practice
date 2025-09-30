# script.py
# Простий е2е-пайплайн: EDA → масштабування → пошук K → KMeans → візуалізація → коротка інтерпретація.
# Виправлення: робота з оригінальними назвами колонок Kaggle (Genre, Annual Income (k$), Spending Score (1-100)),
# надійний шлях до CSV (поруч зі скриптом), безпечний pairplot, вибір K також через silhouette.

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- 1) Завантаження даних ---
here = Path(__file__).resolve().parent
csv_path = here / "Mall_Customers.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        "Файл 'Mall_Customers.csv' не знайдено поруч зі скриптом. "
        "Скачай датасет з Kaggle (Mall Customer Segmentation Data) і поклади біля script.py."
    )

df = pd.read_csv(csv_path)
print(f"Дані завантажено: {csv_path.name}, форма: {df.shape}")

# --- 2) Приведення назв колонок до зручного вигляду ---
# Оригінальний CSV зазвичай має колонки: CustomerID, Genre, Age, Annual Income (k$), Spending Score (1-100)
rename_map = {
    'Annual Income (k$)': 'AnnualIncome',
    'Spending Score (1-100)': 'SpendingScore',
    'Genre': 'Gender'  # стандартизуємо назву для hue
}
# Якщо у когось уже 'Gender' або інші варіанти — errors='ignore'
df = df.rename(columns=rename_map)

required = {'AnnualIncome', 'SpendingScore'}
if not required.issubset(df.columns):
    raise ValueError(
        f"Очікувались колонки {required}, але знайдено {set(df.columns)}. "
        "Перевір правильність датасету."
    )

# --- 3) EDA (коротко) ---
print("\n--- EDA ---")
print(df.head())
print("\nІнфо:")
df.info()
print("\nОписові статистики:")
print(df.describe())

# Безпечний pairplot: якщо є колонка Gender — використаємо hue, інакше без hue.
print("\nБудуємо pairplot (може зайняти трохи часу)...")
plot_df = df.drop(columns=['CustomerID'], errors='ignore').copy()
try:
    if 'Gender' in plot_df.columns:
        sns.pairplot(plot_df, hue='Gender', palette='viridis', corner=True)
    else:
        sns.pairplot(plot_df, palette='viridis', corner=True)
    plt.suptitle('Pairplot клієнтів', y=1.02)
    plt.show()
except Exception as e:
    print(f"Не вдалося побудувати pairplot: {e}")

# --- 4) Підготовка до кластеризації ---
X = df[['AnnualIncome', 'SpendingScore']].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nОзнаки масштабовано (StandardScaler).")

# --- 5) Пошук оптимального K ---
print("\n--- Пошук K (Elbow + Silhouette) ---")
wcss = []
sil_scores = []

k_values = range(2, 11)  # з 2, бо silhouette не визначений для k=1
for k in k_values:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
    sil_scores.append(sil)

# Графік "лікоть"
plt.figure(figsize=(10, 4))
plt.plot(k_values, wcss, marker='o')
plt.xticks(k_values)
plt.title('Метод ліктя (WCSS)')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Графік silhouette
plt.figure(figsize=(10, 4))
plt.plot(k_values, sil_scores, marker='o')
plt.xticks(k_values)
plt.title('Silhouette Score vs K')
plt.xlabel('K')
plt.ylabel('Silhouette')
plt.grid(True)
plt.show()

# Оберемо K як максимум silhouette (часто для цього датасету це 5, але зробимо автоматично)
optimal_k = k_values[int(np.argmax(sil_scores))]
print(f"\nОптимальний K за silhouette: {optimal_k}")

# --- 6) Фінальна модель KMeans ---
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = cluster_labels
print("\nПерші 5 рядків з мітками кластерів:")
print(df.head())

# Центроїди у вихідному масштабі (інверсуємо StandardScaler)
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids, columns=['AnnualIncome', 'SpendingScore'])
print("\nЦентроїди (в оригінальному масштабі):")
print(centroids_df)

# --- 7) Візуалізація кластерів ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df, x='AnnualIncome', y='SpendingScore',
    hue='Cluster', palette='viridis', s=100, alpha=0.9
)
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    s=300, c='red', marker='X', label='Centroids'
)
plt.title(f'KMeans кластери (K={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# --- 8) Коротка інтерпретація ---
summary = (
    df.drop(columns=['CustomerID'], errors='ignore')
      .groupby('Cluster')[['Age', 'AnnualIncome', 'SpendingScore']]
      .mean()
      .round(2)
)
print("\nСередні значення по кластерах:")
print(summary)

print("\nСегментацію завершено.")
