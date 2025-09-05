import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# --- Виправлення: Розширений блок для завантаження ресурсів NLTK ---
# Тепер він перевіряє всі необхідні для цього скрипта пакети
try:
    from nltk.corpus import stopwords

    stopwords.words('english')
except LookupError:
    print("Завантаження ресурсів NLTK (stopwords)...")
    nltk.download('stopwords')

try:
    from nltk.stem import WordNetLemmatizer

    WordNetLemmatizer().lemmatize("test")
except LookupError:
    print("Завантаження ресурсів NLTK (wordnet)...")
    nltk.download('wordnet')

try:
    nltk.word_tokenize("test")
except LookupError:
    print("Завантаження ресурсів NLTK (punkt)...")
    nltk.download('punkt')

# --- ДОДАНО НОВУ ПЕРЕВІРКУ ---
try:
    # Цей рядок імітує помилку, щоб перевірити наявність punkt_tab
    # Хоча це не прямий виклик, помилка виникає всередині NLTK, тому
    # просто завантажимо ресурс, щоб уникнути проблеми.
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Завантаження ресурсів NLTK (punkt_tab)...")
    nltk.download('punkt_tab')

# Тепер можемо імпортувати їх безпечно
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# --- Кінець виправлення ---


# Scikit-learn для ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

print("\nБібліотеки імпортовано, ресурси NLTK перевірено/завантажено.")

# --- 1. Завантаження даних ---
file_path_tsv = 'SMSSpamCollection'
file_path_csv = 'spam.csv'

try:
    df = pd.read_csv(file_path_tsv, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
    print(f"Дані завантажено з '{file_path_tsv}' (роздільник - таб).")
except FileNotFoundError:
    try:
        df = pd.read_csv(file_path_csv, encoding='latin-1')
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']]
            df.columns = ['label', 'message']
            print(f"Дані завантажено з '{file_path_csv}' (роздільник - кома, з заголовком).")
        else:
            df = pd.read_csv(file_path_csv, header=None, names=['label', 'message'], encoding='latin-1')
            print(f"Дані завантажено з '{file_path_csv}' (роздільник - кома, без заголовка).")
    except FileNotFoundError:
        print(f"Помилка: Файли '{file_path_tsv}' та '{file_path_csv}' не знайдено.")
        exit()
except Exception as e:
    print(f"Сталася помилка при читанні файлу: {e}")
    exit()

# --- 2. Початкове дослідження даних (EDA) ---
print("\n--- 2. Початкове дослідження даних (EDA) ---")

print("\nПерші 5 рядків даних:")
print(df.head())

print("\nІнформація про DataFrame:")
df.info()

print("\nРозподіл класів (ham/spam):")
print(df['label'].value_counts())
sns.countplot(data=df, x='label', hue='label', palette='viridis', legend=False)
plt.title('Розподіл повідомлень Ham vs Spam')
plt.show()

df['message_length'] = df['message'].apply(len)
print("\nСтатистика довжини повідомлень:")
print(df['message_length'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='message_length', hue='label', kde=True, bins=50)
plt.title('Розподіл довжини повідомлень для Ham та Spam')
plt.xlabel('Довжина повідомлення (символи)')
plt.show()

# --- 3. Передобробка Тексту ---
print("\n--- 3. Передобробка Тексту ---")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    words = nltk.word_tokenize(text)
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)


print("Застосування передобробки до повідомлень...")
df['cleaned_message'] = df['message'].apply(preprocess_text)
print("Передобробку завершено.")

print("\nПриклад повідомлення до та після обробки:")
print("Оригінал:", df['message'][0])
print("Очищене:", df['cleaned_message'][0])

# --- 4. Підготовка даних для Моделювання ---
print("\n--- 4. Підготовка даних для Моделювання ---")

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print("Створено числову колонку 'label_num' (ham=0, spam=1).")

X = df['cleaned_message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nРозмір тренувального набору: {len(X_train)}")
print(f"Розмір тестового набору: {len(X_test)}")

# --- 5. Створення та Тренування Моделей (з Пайплайнами) ---
print("\n--- 5. Створення та Тренування Моделей ---")

models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
    "Support Vector Machine (SVC)": SVC(kernel='linear', random_state=42, probability=True)
}

results = {}

for name, model in models.items():
    print(f"\n--- Тренування моделі: {name} ---")
    text_clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', model),
    ])
    text_clf_pipeline.fit(X_train, y_train)
    y_pred = text_clf_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    results[name] = {'accuracy': accuracy, 'report': report, 'conf_matrix': conf_matrix}
    print(f"Результати для {name}:")
    print(f"Точність (Accuracy): {accuracy:.4f}")
    print("Звіт про класифікацію:")
    print(report)
    print("Матриця помилок:")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Ham', 'Spam'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Матриця помилок для {name}')
    plt.show()

# --- 6. Порівняння результатів ---
print("\n--- 6. Порівняння результатів ---")
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [res['accuracy'] for res in results.values()]
})
results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print("\nПорівняльна таблиця точності моделей:")
print(results_df)

print("\n--- Класифікацію спаму завершено ---")
