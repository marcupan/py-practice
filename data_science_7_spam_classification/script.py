import numpy as np
import string
import logging
import sys
import math
from pathlib import Path
import argparse

# Scikit-learn імпорти будуть виконані всередині main() з безпечними перевірками середовища

logger = logging.getLogger(__name__)

def main():
    # --- CLI аргументи ---
    parser = argparse.ArgumentParser(description="SMS Spam Classification")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Шлях до каталогу з даними або до файлу SMSSpamCollection/spam.csv. Якщо не вказано — шукаємо в ./data/ та поруч зі скриптом.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Спробувати імпортувати matplotlib/seaborn і показувати графіки (за замовчуванням вимкнено, щоб уникати NumPy/Matplotlib ABI-конфліктів).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Бібліотеки імпортовано.")

    # Environment checks and safe import of pandas
    # Soft-check NumPy version: warn if older than versions commonly required by new pandas, but do not exit.
    try:
        np_version = tuple(int(x) for x in np.__version__.split('.')[:3])
    except Exception:
        np_version = (0, 0, 0)
    required_np = (1, 22, 4)
    if np_version < required_np:
        logger.warning(
            "Detected NumPy version %s (<%s). Proceeding to try importing pandas. If pandas import fails due to a version\n"
            "mismatch, please either upgrade NumPy (pip install -U \"numpy>=1.22.4\") or install a pandas version\n"
            "compatible with your NumPy (e.g., pip install \"pandas==1.3.5\").",
            np.__version__, '.'.join(map(str, required_np))
        )

    pandas_available = True
    try:
        import pandas as pd  # noqa: F401
    except Exception as e:
        pandas_available = False
        logger.warning(
            "Pandas is unavailable and will be skipped. This is often due to a NumPy/Pandas version mismatch. Suggested fixes:\n"
            "- Upgrade NumPy: pip install -U \"numpy>=1.22.4\"\n"
            "- Or install a pandas version compatible with your NumPy (e.g., pip install \"pandas==1.3.5\")\n"
            "If using conda: conda install pandas numpy\nOriginal import error: %s",
            e,
        )

    # Optional plotting libraries are imported ONLY if --plots передано.
    plotting_enabled = False
    if pandas_available and args.plots:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            import seaborn as sns  # noqa: F401
            plotting_enabled = True
        except Exception as e:
            plotting_enabled = False
            logger.warning("Plotting libraries unavailable or failed to import (possibly due to pandas/NumPy deps). Skipping plots. Error: %s", e)
    else:
        if args.plots and not pandas_available:
            logger.info("Pandas not available; skipping plotting imports and all plots.")
        elif not args.plots:
            logger.info("Прапорець --plots не заданий — імпорт графічних бібліотек вимкнено.")

    # Import scikit-learn lazily with environment guard to avoid NumPy/SciPy ABI crashes at import time
    sklearn_available = True
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        if plotting_enabled:
            from sklearn.metrics import ConfusionMatrixDisplay
    except Exception as e:
        sklearn_available = False
        logger.error(
            "Failed to import scikit-learn (or its SciPy/NumPy deps). This usually indicates a binary incompatibility between installed packages. We'll run a simplified built-in fallback model instead.\n"
            "Suggested fixes:\n"
            "- Upgrade NumPy/SciPy/scikit-learn together:\n    pip install -U --force-reinstall numpy scipy scikit-learn\n"
            "- Or install versions compatible with your current NumPy (%s). For example:\n    pip install \"scikit-learn==1.0.2\" \"scipy==1.7.3\"\n"
            "If using conda:\n    conda install -c conda-forge numpy scipy scikit-learn\n"
            "Original import error: %s",
            np.__version__, e,
        )

    # --- 1. Надійний пошук даних ---
    script_dir = Path(__file__).resolve().parent
    # Якщо передали файл напряму
    candidates = []
    if args.data:
        p = Path(args.data).expanduser().resolve()
        if p.is_file():
            candidates = [p]
        else:
            # передано каталог
            candidates = [p / "SMSSpamCollection", p / "spam.csv"]
    else:
        data_dir = script_dir / "data"
        candidates = [
            data_dir / "SMSSpamCollection",
            data_dir / "spam.csv",
            script_dir / "SMSSpamCollection",
            script_dir / "spam.csv",
        ]

    file_path = next((c for c in candidates if c.exists()), None)
    if not file_path:
        logger.error("Помилка: Дані не знайдено. Шукали в: %s", ", ".join(map(str, candidates)))
        sys.exit(1)

    # --- 1b. Завантаження даних ---
    if pandas_available:
        try:
            if file_path.name == "SMSSpamCollection":
                df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
                logger.info(f"Дані завантажено з '{file_path}' (таб-роздільник).")
            else:
                df = pd.read_csv(file_path, encoding='latin-1')
                if {'v1', 'v2'}.issubset(df.columns):
                    df = df[['v1', 'v2']]
                    df.columns = ['label', 'message']
                else:
                    # Якщо інші заголовки — беремо перші дві колонки
                    if df.shape[1] >= 2:
                        df = df.iloc[:, :2]
                        df.columns = ['label', 'message']
                logger.info(f"Дані завантажено з '{file_path}'.")
        except Exception as e:
            logger.exception(f"Сталася помилка при читанні файлу: {e}")
            sys.exit(1)
    else:
        # Fallback CSV/TSV loading without pandas
        import csv
        labels = []
        messages = []
        try:
            if file_path.name == "SMSSpamCollection":
                with open(file_path, 'r', encoding='latin-1', newline='') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        if len(row) >= 2:
                            labels.append(row[0]); messages.append(row[1])
                logger.info(f"Дані завантажено з '{file_path}' (таб) без pandas.")
            else:
                with open(file_path, 'r', encoding='latin-1', newline='') as f:
                    sample = f.read(2048); f.seek(0)
                    has_header = csv.Sniffer().has_header(sample)
                    if has_header:
                        dr = csv.DictReader(f)
                        fieldnames = [fn.strip() for fn in (dr.fieldnames or [])]
                        if 'v1' in fieldnames and 'v2' in fieldnames:
                            for row in dr:
                                labels.append(row.get('v1', '')); messages.append(row.get('v2', ''))
                        else:
                            first_two = fieldnames[:2]
                            for row in dr:
                                labels.append(row.get(first_two[0], '')); messages.append(row.get(first_two[1], ''))
                        logger.info(f"Дані завантажено з '{file_path}' (кома, з заголовком) без pandas.")
                    else:
                        f.seek(0)
                        rr = csv.reader(f)
                        first_row = next(rr, None)
                        if first_row is not None:
                            if len(first_row) >= 2 and first_row[0].strip().lower() == 'v1' and first_row[1].strip().lower() == 'v2':
                                logger.info("Виявлено рядок заголовка ['v1','v2'] у файлі без заголовка — пропускаємо його.")
                            else:
                                if len(first_row) >= 2:
                                    labels.append(first_row[0]); messages.append(first_row[1])
                        for row in rr:
                            if len(row) >= 2:
                                labels.append(row[0]); messages.append(row[1])
                        logger.info(f"Дані завантажено з '{file_path}' (кома, без заголовка) без pandas.")
        except Exception as e:
            logger.exception(f"Сталася помилка при читанні файлу (csv/tsv без pandas): {e}")
            sys.exit(1)

    # --- 2. Початкове дослідження даних (EDA) ---
    logger.info("--- 2. Початкове дослідження даних (EDA) ---")

    if pandas_available:
        logger.info("Перші 5 рядків даних:\n%s", df.head())

        logger.info("Інформація про DataFrame:")
        df.info()

        logger.info("Розподіл класів (ham/spam):\n%s", df['label'].value_counts())
        if plotting_enabled:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.countplot(data=df, x='label', hue='label', palette='viridis', legend=False)
            plt.title('Розподіл повідомлень Ham vs Spam')
            plt.show()
        else:
            logger.info("Пропускаємо графік розподілу класів (plotting disabled).")

        df['message_length'] = df['message'].apply(len)
        logger.info("Статистика довжини повідомлень:\n%s", df['message_length'].describe())

        if plotting_enabled:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='message_length', hue='label', kde=True, bins=50)
            plt.title('Розподіл довжини повідомлень для Ham та Spam')
            plt.xlabel('Довжина повідомлення (символи)')
            plt.show()
        else:
            logger.info("Пропускаємо гістограму довжин повідомлень (plotting disabled).")
    else:
        # Fallback EDA without pandas
        from collections import Counter
        logger.info("Перші 5 рядків даних (без pandas):")
        for i in range(min(5, len(messages))):
            logger.info("%d: label=%s | message=%s", i, labels[i], messages[i])
        counts = Counter(labels)
        logger.info("Розподіл класів (ham/spam): %s", counts)
        msg_lengths = [len(m) for m in messages]
        if len(msg_lengths) == 0:
            logger.warning("Порожній набір даних для EDA.")
        else:
            arr = np.array(msg_lengths)
            stats = {
                'count': arr.size,
                'mean': float(arr.mean()),
                'std': float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                'min': int(arr.min()),
                '25%': float(np.percentile(arr, 25)),
                '50%': float(np.percentile(arr, 50)),
                '75%': float(np.percentile(arr, 75)),
                'max': int(arr.max()),
            }
            logger.info("Статистика довжини повідомлень (без pandas): %s", stats)
        logger.info("Пропускаємо всі графіки (pandas/plotting недоступні).")

    # --- 3. Передобробка Тексту ---
    logger.info("--- 3. Передобробка Тексту ---")

    # Використовуємо легку обробку stdlib (пунктуація/регістр), а стоп-слова та токенізацію виконає TfidfVectorizer
    def simple_clean(text):
        if not isinstance(text, str):
            return ""
        cleaned = ''.join(ch for ch in text if ch not in string.punctuation)
        return cleaned.lower().strip()

    logger.info("Застосування легкої stdlib-обробки (пунктуація/регістр)...")
    if pandas_available:
        df['cleaned_message'] = df['message'].apply(simple_clean)
        logger.info("Передобробку завершено.")
        try:
            logger.info("Приклад повідомлення до та після обробки:\nОригінал: %s\nОчищене: %s", df['message'][0], df['cleaned_message'][0])
        except Exception:
            pass
    else:
        cleaned_messages = [simple_clean(m) for m in messages]
        logger.info("Передобробку завершено (без pandas).")
        if len(messages) > 0:
            logger.info("Приклад повідомлення до та після обробки:\nОригінал: %s\nОчищене: %s", messages[0], cleaned_messages[0])

    # --- 4. Підготовка даних для Моделювання ---
    logger.info("--- 4. Підготовка даних для Моделювання ---")

    if pandas_available:
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        logger.info("Створено числову колонку 'label_num' (ham=0, spam=1).")
        X = df['cleaned_message']
        y = df['label_num']
    else:
        label_map = {'ham': 0, 'spam': 1}
        y = [label_map.get(lbl, 0) for lbl in labels]
        X = cleaned_messages
        logger.info("Побудовано цільовий вектор без pandas (ham=0, spam=1).")

    # Розбиття на train/test
    if sklearn_available:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # Власна стратифікована вибірка 80/20 без sklearn
        try:
            y_list = list(y) if not isinstance(y, list) else y
            idx_ham = [i for i, t in enumerate(y_list) if int(t) == 0]
            idx_spam = [i for i, t in enumerate(y_list) if int(t) == 1]
            rng = np.random.RandomState(42)
            rng.shuffle(idx_ham); rng.shuffle(idx_spam)
            def split_indices(idxs):
                k = max(1, int(round(0.2 * len(idxs)))) if len(idxs) > 0 else 0
                return idxs[k:], idxs[:k]
            train_ham, test_ham = split_indices(idx_ham)
            train_spam, test_spam = split_indices(idx_spam)
            train_idx = train_ham + train_spam
            test_idx = test_ham + test_spam
            rng.shuffle(train_idx); rng.shuffle(test_idx)
            def take(seq, indices):
                if hasattr(seq, 'iloc'):
                    return list(seq.iloc[indices])
                elif hasattr(seq, '__getitem__'):
                    return [seq[i] for i in indices]
                else:
                    return [seq for _ in indices]
            X_train = take(X, train_idx); X_test = take(X, test_idx)
            y_train = [y_list[i] for i in train_idx]; y_test = [y_list[i] for i in test_idx]
        except Exception as e:
            logger.exception("Не вдалося виконати власне розбиття train/test: %s", e)
            sys.exit(2)

    logger.info("Розмір тренувального набору: %d", len(X_train))
    logger.info("Розмір тестового набору: %d", len(X_test))

    # --- 5. Створення та Тренування Моделей (з Пайплайнами) ---
    logger.info("--- 5. Створення та Тренування Моделей ---")

    results = {}

    if sklearn_available:
        models = {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
            "Support Vector Machine (SVC)": SVC(kernel='linear', random_state=42, probability=True)
        }
        for name, model in models.items():
            logger.info("--- Тренування моделі: %s ---", name)
            text_clf_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
                ('clf', model),
            ])
            text_clf_pipeline.fit(X_train, y_train)
            y_pred = text_clf_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'])
            conf_matrix = confusion_matrix(y_test, y_pred)
            results[name] = {'accuracy': accuracy, 'report': report, 'conf_matrix': conf_matrix}
            logger.info("Результати для %s:", name)
            logger.info("Точність (Accuracy): %.4f", accuracy)
            logger.info("Звіт про класифікацію:\n%s", report)
            logger.info("Матриця помилок:\n%s", conf_matrix)
            if plotting_enabled:
                import matplotlib.pyplot as plt
                from sklearn.metrics import ConfusionMatrixDisplay
                disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Ham', 'Spam'])
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f'Матриця помилок для {name}')
                plt.show()
            else:
                logger.info("Пропускаємо відображення матриці помилок для %s (plotting disabled).", name)
    else:
        # Проста резервна реалізація Multinomial Naive Bayes без sklearn
        logger.info("Скикит-лерн недоступний — запускаємо спрощений вбудований класифікатор Multinomial Naive Bayes.")
        def tokenize(text):
            return [t for t in str(text).split() if t]
        vocab = {}
        def add_to_vocab(tokens):
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        for msg, lbl in zip(X_train, y_train):
            add_to_vocab(tokenize(msg))
        V = len(vocab)
        alpha = 1.0  # Лапласове згладжування
        total_words = {0: 0, 1: 0}
        word_counts = {0: np.zeros(V, dtype=np.int64), 1: np.zeros(V, dtype=np.int64)}
        class_counts = {0: 0, 1: 0}
        for msg, lbl in zip(X_train, y_train):
            c = int(lbl)
            class_counts[c] += 1
            for tok in tokenize(msg):
                j = vocab.get(tok)
                if j is not None:
                    word_counts[c][j] += 1
                    total_words[c] += 1
        n_train = len(y_train)
        priors = {0: (class_counts[0] / n_train if n_train else 0.5), 1: (class_counts[1] / n_train if n_train else 0.5)}
        log_probs = {0: None, 1: None}
        default_log_prob = {}
        for c in [0, 1]:
            denom = total_words[c] + alpha * V if V > 0 else 1.0
            probs_c = (word_counts[c] + alpha) / denom
            probs_c = np.maximum(probs_c, 1e-12)
            log_probs[c] = np.log(probs_c)
            default_log_prob[c] = math.log(alpha / denom) if denom > 0 else math.log(1e-12)
        log_prior = {0: math.log(max(priors[0], 1e-12)), 1: math.log(max(priors[1], 1e-12))}
        def predict_one(msg):
            lp0 = log_prior[0]; lp1 = log_prior[1]
            for tok in tokenize(msg):
                j = vocab.get(tok)
                if j is None:
                    lp0 += default_log_prob[0]; lp1 += default_log_prob[1]
                else:
                    lp0 += float(log_probs[0][j]); lp1 += float(log_probs[1][j])
            return 1 if lp1 > lp0 else 0
        y_pred = [predict_one(msg) for msg in X_test]
        def confusion(y_true, y_hat):
            tn = fp = fn = tp = 0
            for yt, yp in zip(y_true, y_hat):
                if int(yt) == 1 and int(yp) == 1: tp += 1
                elif int(yt) == 0 and int(yp) == 0: tn += 1
                elif int(yt) == 0 and int(yp) == 1: fp += 1
                else: fn += 1
            return np.array([[tn, fp], [fn, tp]], dtype=int)
        conf_matrix = confusion(y_test, y_pred)
        accuracy = float((conf_matrix[0,0] + conf_matrix[1,1]) / max(1, conf_matrix.sum()))
        def pr_re_f1(cm, pos):
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            if pos == 1:
                p = tp / max(1, tp + fp)
                r = tp / max(1, tp + fn)
            else:
                p = tn / max(1, tn + fn)
                r = tn / max(1, tn + fp)
            f1 = 2 * p * r / max(1e-12, p + r)
            return p, r, f1
        p0, r0, f10 = pr_re_f1(conf_matrix, 0)
        p1, r1, f11 = pr_re_f1(conf_matrix, 1)
        report = (
            f"Classification report (fallback)\n"
            f"Ham (0): precision={p0:.3f} recall={r0:.3f} f1-score={f10:.3f}\n"
            f"Spam (1): precision={p1:.3f} recall={r1:.3f} f1-score={f11:.3f}\n"
            f"Accuracy={accuracy:.4f}"
        )
        results["Simple MultinomialNB (fallback)"] = {
            'accuracy': accuracy,
            'report': report,
            'conf_matrix': conf_matrix,
        }
        logger.info("Результати (fallback): Точність (Accuracy): %.4f", accuracy)
        logger.info("Матриця помилок (fallback):\n%s", conf_matrix)
        logger.info("Звіт про класифікацію (fallback):\n%s", report)

    # --- 6. Порівняння результатів ---
    logger.info("--- 6. Порівняння результатів ---")
    if pandas_available:
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [res['accuracy'] for res in results.values()]
        })
        results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        logger.info("Порівняльна таблиця точності моделей:\n%s", results_df)
    else:
        # Fallback: sort and log without DataFrame
        sorted_items = sorted(results.items(), key=lambda kv: kv[1]['accuracy'], reverse=True)
        logger.info("Порівняльна таблиця точності моделей (без pandas):")
        for name, res in sorted_items:
            logger.info("%s: Accuracy=%.4f", name, res['accuracy'])

    logger.info("--- Класифікацію спаму завершено ---")


if __name__ == "__main__":
    main()
