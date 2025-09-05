import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Модулі Scikit-learn
from sklearn.datasets import fetch_california_housing  # Для завантаження датасету
from sklearn.model_selection import train_test_split  # Для розділення даних
from sklearn.preprocessing import StandardScaler  # Для масштабування ознак
from sklearn.pipeline import Pipeline  # Для об'єднання кроків

# Моделі регресії
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor # Ще одна потужна модель (опціонально)

# Метрики для оцінки регресії
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Бібліотеки імпортовано.")

# --- 1. Завантаження даних ---
print("\n--- 1. Завантаження даних California Housing ---")
# fetch_california_housing повертає об'єкт Bunch, схожий на словник
california_housing = fetch_california_housing(as_frame=True)

# Створюємо pandas DataFrame
df = california_housing.frame
# Цільова змінна (median house value) вже знаходиться в df під назвою 'MedHouseVal'
# df['MedHouseVal'] = california_housing.target # Цей рядок не потрібен, якщо as_frame=True

print(f"Дані завантажено. Розмір: {df.shape}")
print("Ознаки:", california_housing.feature_names)
print("Цільова змінна: MedHouseVal (Median House Value in $100,000s)")

# --- 2. Початкове дослідження даних (EDA) ---
print("\n--- 2. Початкове дослідження даних (EDA) ---")

print("\nПерші 5 рядків даних:")
print(df.head())

print("\nІнформація про DataFrame:")
df.info()
# Зазвичай цей датасет не має пропущених значень

print("\nОписові статистики:")
print(df.describe())

# Візуалізація розподілу цільової змінної (MedHouseVal)
plt.figure(figsize=(10, 6))
sns.histplot(df['MedHouseVal'], kde=True, bins=30)
plt.title('Розподіл Медіанної Вартості Житла (MedHouseVal)')
plt.xlabel('Медіанна Вартість ($100,000s)')
plt.ylabel('Кількість районів')
plt.tight_layout()
plt.show()

# Візуалізація розподілу ключових ознак (наприклад, MedInc - медіанний дохід)
plt.figure(figsize=(10, 6))
sns.histplot(df['MedInc'], kde=True, bins=30)
plt.title('Розподіл Медіанного Доходу (MedInc)')
plt.xlabel('Медіанний Дохід ($10,000s)')
plt.ylabel('Кількість районів')
plt.tight_layout()
plt.show()

# Діаграма розсіювання: Медіанний дохід vs Вартість житла
plt.figure(figsize=(10, 6))
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.3)
plt.title('Вартість житла vs Медіанний дохід')
plt.xlabel('Медіанний Дохід ($10,000s)')
plt.ylabel('Медіанна Вартість ($100,000s)')
plt.tight_layout()
plt.show()

# Теплова карта кореляцій
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Матриця кореляцій ознак')
plt.tight_layout()
plt.show()
# Бачимо сильну позитивну кореляцію MedInc з MedHouseVal

# --- 3. Підготовка даних до моделювання ---
print("\n--- 3. Підготовка даних до моделювання ---")

# Визначаємо ознаки (X) та цільову змінну (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Розділяємо дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# test_size=0.2 -> 20% даних для тестування
# random_state=42 -> для відтворюваності

print(f"Розмір тренувального набору: {X_train.shape}, {y_train.shape}")
print(f"Розмір тестового набору: {X_test.shape}, {y_test.shape}")

# Масштабування числових ознак є важливим для лінійних моделей, Ridge, Lasso
# StandardScaler приводить дані до середнього 0 та стандартного відхилення 1
# Ми будемо використовувати його всередині Пайплайнів

# --- 4. Визначення, Тренування та Оцінка Моделей ---
print("\n--- 4. Тренування та Оцінка Моделей ---")

# Визначимо моделі регресії
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),  # alpha - параметр регуляризації
    "Lasso Regression": Lasso(alpha=0.01, random_state=42, max_iter=2000),  # alpha - параметр регуляризації
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # n_jobs=-1 -> використовувати всі ядра процесора
    # "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42) # Опціонально
}

results = {}  # Словник для зберігання результатів

# Цикл для тренування та оцінки кожної моделі
for name, model in models.items():
    print(f"\n--- Тренування моделі: {name} ---")
    # Створюємо пайплайн: Масштабування + Модель
    # Це гарантує, що масштабування застосовується коректно (навчається на train, трансформує train і test)
    pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                               ('regressor', model)])

    # Тренуємо пайплайн
    pipeline.fit(X_train, y_train)

    # Робимо прогнози на тестових даних
    y_pred = pipeline.predict(X_test)

    # Оцінюємо модель за допомогою регресійних метрик
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Корінь з MSE
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    # Виводимо результати
    print(f"Результати для {name}:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")  # Середня абсолютна помилка
    print(f"Mean Squared Error (MSE): {mse:.4f}")  # Середня квадратична помилка
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")  # Корінь з середньої квадратичної помилки
    print(f"R-squared (R2): {r2:.4f}")  # Коефіцієнт детермінації

# --- 5. Порівняння результатів ---
print("\n--- 5. Порівняння результатів ---")
results_df = pd.DataFrame(results).T  # Транспонуємо для зручного вигляду
results_df = results_df.sort_values(by='R2', ascending=False)  # Сортуємо за R2 (чим ближче до 1, тим краще)
print(results_df)

# Візуалізація порівняння (наприклад, за R2)
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df['R2'], y=results_df.index, palette='cubehelix')
plt.title('Порівняння моделей за R-squared (R2)')
plt.xlabel('R-squared (Коефіцієнт детермінації)')
plt.ylabel('Модель')
plt.xlim(0, 1.0)  # R2 зазвичай між 0 і 1 для хороших моделей
plt.tight_layout()
plt.show()

# Візуалізація порівняння (наприклад, за RMSE)
plt.figure(figsize=(10, 6))
results_df_rmse = results_df.sort_values(by='RMSE', ascending=True)  # Сортуємо за RMSE (чим менше, тим краще)
sns.barplot(x=results_df_rmse['RMSE'], y=results_df_rmse.index, palette='rocket')
plt.title('Порівняння моделей за RMSE')
plt.xlabel('Root Mean Squared Error (RMSE)')
plt.ylabel('Модель')
plt.tight_layout()
plt.show()

# --- 6. Аналіз найкращої моделі (Приклад: Random Forest) ---
print("\n--- 6. Аналіз найкращої моделі (Random Forest) ---")
best_model_name = results_df.index[0]  # Назва найкращої моделі за R2
print(f"Найкраща модель (за R2): {best_model_name}")

# Отримаємо пайплайн найкращої моделі
best_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                ('regressor', models[best_model_name])])
best_pipeline.fit(X_train, y_train)  # Перенавчаємо (хоча вже навчили в циклі)
y_pred_best = best_pipeline.predict(X_test)

# Візуалізація: Справжні значення vs Передбачені значення
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_best, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)  # Ідеальна лінія y=x
plt.title(f'Справжні vs Передбачені значення ({best_model_name})')
plt.xlabel('Справжня вартість ($100,000s)')
plt.ylabel('Передбачена вартість ($100,000s)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Аналіз важливості ознак (для моделей на основі дерев)
if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
    try:
        importances = best_pipeline.named_steps['regressor'].feature_importances_
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='Spectral')
        plt.title(f'Важливість ознак для {best_model_name}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Не вдалося побудувати графік важливості ознак: {e}")

print("\n--- Прогнозування вартості житла завершено ---")
