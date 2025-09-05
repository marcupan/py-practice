import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re  # Для роботи з регулярними виразами (витягнення титулів)

# Модулі Scikit-learn
from sklearn.model_selection import train_test_split  # Для розділення даних
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Для масштабування та кодування
from sklearn.impute import SimpleImputer  # Для заповнення пропусків (альтернатива pandas)
from sklearn.compose import ColumnTransformer  # Для застосування різних перетворень до різних колонок
from sklearn.pipeline import Pipeline  # Для об'єднання кроків передобробки та моделювання

# Моделі
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Метрики для оцінки
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

print("Бібліотеки імпортовано.")

# --- 1. Завантаження даних ---
file_path = 'train.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Дані успішно завантажено з '{file_path}'. Розмір: {df.shape}")
except FileNotFoundError:
    print(f"Помилка: Файл '{file_path}' не знайдено.")
    exit()

# Зробимо копію для безпеки
df_processed = df.copy()

# --- 2. Інженерія Ознак та Передобробка ---
print("\n--- 2. Інженерія Ознак та Передобробка ---")

# 2.1 Обробка пропущених значень
# Age: Заповнимо медіаною
median_age = df_processed['Age'].median()
df_processed['Age'].fillna(median_age, inplace=True)
print(f"NaN в 'Age' заповнено медіаною ({median_age:.2f}).")

# Embarked: Заповнимо модою
mode_embarked = df_processed['Embarked'].mode()[0]
df_processed['Embarked'].fillna(mode_embarked, inplace=True)
print(f"NaN в 'Embarked' заповнено модою ('{mode_embarked}').")

# Cabin: Створимо бінарну ознаку 'Has_Cabin'
df_processed['Has_Cabin'] = df_processed['Cabin'].notna().astype(int)  # 1 якщо є каюта, 0 - якщо NaN
print("Створено бінарну ознаку 'Has_Cabin'.")

# 2.2 Створення нових ознак
# FamilySize
df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
print("Створено ознаку 'FamilySize'.")

# IsAlone
df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
print("Створено бінарну ознаку 'IsAlone'.")


# Title (витягнення титулу з імені)
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


df_processed['Title'] = df_processed['Name'].apply(get_title)
# Замінимо рідкісні титули на 'Rare' або об'єднаємо їх
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')
print("Створено та оброблено ознаку 'Title':")
print(df_processed['Title'].value_counts())

# 2.3 Видалення непотрібних колонок
# Видаляємо оригінальні колонки, які були оброблені або не потрібні для моделі
df_processed.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)
print("Видалено непотрібні колонки: 'PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'.")

# 2.4 Визначення типів колонок для передобробки
target = 'Survived'
numerical_features = ['Age', 'Fare', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Has_Cabin',
                        'IsAlone']  # Pclass розглядаємо як категоріальну

print(f"\nЦільова змінна: {target}")
print(f"Числові ознаки: {numerical_features}")
print(f"Категоріальні ознаки: {categorical_features}")

# --- 3. Розділення даних на Тренувальний та Валідаційний набори ---
print("\n--- 3. Розділення даних ---")
X = df_processed.drop(target, axis=1)  # Ознаки
y = df_processed[target]  # Цільова змінна

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# test_size=0.2 -> 20% даних піде на валідацію
# random_state=42 -> для відтворюваності результатів
# stratify=y -> зберігає пропорцію класів (виживших/загиблих) в обох наборах

print(f"Розмір тренувального набору: {X_train.shape}, {y_train.shape}")
print(f"Розмір валідаційного набору: {X_val.shape}, {y_val.shape}")

# --- 4. Створення Пайплайнів Передобробки ---
print("\n--- 4. Створення Пайплайнів Передобробки ---")

# Пайплайн для числових ознак: заповнення пропусків (якщо раптом з'явились) + масштабування
# Хоча ми вже заповнили Age, додамо SimpleImputer для надійності
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Заповнення медіаною
    ('scaler', StandardScaler())  # Масштабування
])

# Пайплайн для категоріальних ознак: заповнення пропусків (якщо є) + One-Hot Encoding
# handle_unknown='ignore' допомагає уникнути помилок, якщо в валідаційних даних з'явиться категорія, якої не було в тренувальних
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Заповнення модою
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-Hot кодування
])

# Об'єднуємо пайплайни за допомогою ColumnTransformer
# Він застосує відповідний пайплайн до відповідних колонок
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Залишити інші колонки (якщо є) без змін
)

print("Пайплайни для числових та категоріальних даних створено.")

# --- 5. Визначення, Тренування та Оцінка Моделей ---
print("\n--- 5. Тренування та Оцінка Моделей ---")

# Визначимо моделі, які будемо тестувати
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),  # n_neighbors - гіперпараметр
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)  # n_estimators - гіперпараметр
}

results = {}  # Словник для зберігання результатів

# Цикл для тренування та оцінки кожної моделі
for name, model in models.items():
    print(f"\n--- Тренування моделі: {name} ---")
    # Створюємо повний пайплайн: Передобробка + Модель
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Тренуємо пайплайн на тренувальних даних
    pipeline.fit(X_train, y_train)

    # Робимо прогнози на валідаційних даних
    y_pred = pipeline.predict(X_val)
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]  # Ймовірності для AUC

    # Оцінюємо модель
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    report = classification_report(y_val, y_pred, target_names=['Загинув (0)', 'Вижив (1)'])
    conf_matrix = confusion_matrix(y_val, y_pred)

    results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc, 'report': report, 'conf_matrix': conf_matrix}

    # Виводимо результати
    print(f"Результати для {name}:")
    print(f"Точність (Accuracy): {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("Звіт про класифікацію:")
    print(report)
    print("Матриця помилок:")
    # Візуалізація матриці помилок
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Загинув', 'Вижив'],
                yticklabels=['Загинув', 'Вижив'])
    plt.xlabel('Передбачений клас')
    plt.ylabel('Істинний клас')
    plt.title(f'Матриця помилок для {name}')
    plt.show()

# --- 6. Порівняння результатів ---
print("\n--- 6. Порівняння результатів ---")
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [res['accuracy'] for res in results.values()],
    'ROC AUC': [res['roc_auc'] for res in results.values()]
})
results_df = results_df.sort_values(by='ROC AUC', ascending=False).reset_index(drop=True)
print(results_df)

plt.figure(figsize=(10, 5))
sns.barplot(x='ROC AUC', y='Model', data=results_df, palette='viridis')
plt.title('Порівняння моделей за ROC AUC')
plt.xlabel('ROC AUC Score')
plt.ylabel('Модель')
plt.xlim(0.5, 1.0)  # Встановимо межі для кращої візуалізації
plt.tight_layout()
plt.show()

print("\n--- Прогнозування виживання на 'Титаніку' завершено ---")
