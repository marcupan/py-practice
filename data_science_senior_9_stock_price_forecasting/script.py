import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings

warnings.filterwarnings("ignore")

print("Бібліотеки імпортовано.")

# --- 1. Завантаження даних ---
ticker = 'AAPL'
start_date = '2018-01-01'
end_date = '2024-12-31'

print(f"\n--- 1. Завантаження даних для {ticker} з {start_date} по {end_date} ---")
try:
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError("Завантажено порожній DataFrame. Перевірте тікер або період.")
    print(f"Дані завантажено. Розмір: {df.shape}")
except Exception as e:
    print(f"Помилка завантаження даних: {e}")
    exit()

ts_data = df['Close'].copy()

# --- 2. Дослідження даних (EDA) та Перевірка на стаціонарність ---
# ... (цей блок залишається без змін) ...
print("\n--- 2. EDA та Перевірка на стаціонарність ---")
plt.figure(figsize=(14, 7))
ts_data.plot()
plt.title(f'Ціна закриття акцій {ticker}')
plt.xlabel('Дата')
plt.ylabel('Ціна закриття (USD)')
plt.grid(True)
plt.show()


def check_stationarity(timeseries):
    print('\nРезультати тесту Дікі-Фуллера:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    if dftest[1] <= 0.05:
        print("=> Результат: Ряд є стаціонарним (відхиляємо нульову гіпотезу)")
        return True
    else:
        print("=> Результат: Ряд НЕ є стаціонарним (не вдалося відхилити нульову гіпотезу)")
        return False


check_stationarity(ts_data)

# --- 3. Передобробка: Досягнення стаціонарності (Диференціювання) ---
# ... (цей блок залишається без змін) ...
ts_diff = ts_data.diff().dropna()
print("\n--- 3. Передобробка: Диференціювання ---")
plt.figure(figsize=(14, 7))
ts_diff.plot()
plt.title(f'Різниця цін закриття акцій {ticker} (1 порядок)')
plt.xlabel('Дата')
plt.ylabel('Різниця ціни')
plt.grid(True)
plt.show()
check_stationarity(ts_diff)
d = 1

# --- 4. Визначення параметрів ARIMA (p, q) ---
# ... (цей блок залишається без змін) ...
print("\n--- 4. Аналіз ACF та PACF для визначення p та q ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(ts_diff, ax=axes[0], lags=40)
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(ts_diff, ax=axes[1], lags=40, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()
p = 5
q = 5
print(f"Обрані параметри для ARIMA: p={p}, d={d}, q={q}")

# --- 5. Розділення даних на Тренувальний та Тестовий набори ---
# ... (цей блок залишається без змін) ...
train_size = int(len(ts_data) * 0.8)
train_data, test_data = ts_data[0:train_size], ts_data[train_size:len(ts_data)]
print("\n--- 5. Розділення даних ---")
print(f"Розмір тренувального набору: {len(train_data)}")
print(f"Розмір тестового набору: {len(test_data)}")
print(f"Дата початку тестового набору: {test_data.index.min().date()}")
print(f"Дата кінця тестового набору: {test_data.index.max().date()}")

# --- 6. Побудова та Тренування моделі ARIMA ---
# ... (цей блок залишається без змін) ...
print("\n--- 6. Побудова та Тренування моделі ARIMA ---")
model = ARIMA(train_data, order=(p, d, q))
model_fit = model.fit()
print("\nЗведення моделі ARIMA:")
print(model_fit.summary())

# --- 7. Прогнозування та Оцінка ---
print("\n--- 7. Прогнозування на тестовому наборі та Оцінка ---")

n_forecast_steps = len(test_data)
predictions = model_fit.forecast(steps=n_forecast_steps)

print(f"\nЗроблено прогноз для {len(predictions)} кроків.")

# ВИПРАВЛЕННЯ ТУТ:
# Створюємо DataFrame, використовуючи .values для гарантії одновимірності
# та передаємо індекс тестових даних для правильного вирівнювання
comparison_df = pd.DataFrame(
    {'Actual': test_data.values, 'Predicted': predictions.values},
    index=test_data.index
)
print("\nПорівняння перших 5 реальних та прогнозованих значень:")
print(comparison_df.head())

# Подальші розрахунки метрик тепер працюватимуть коректно
mae = mean_absolute_error(test_data, predictions)
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


mape = mean_absolute_percentage_error(test_data, predictions)

print(f"\nОцінка моделі ARIMA({p},{d},{q}) на тестових даних:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# --- 8. Візуалізація результатів прогнозування ---
# ... (цей блок залишається без змін) ...
print("\n--- 8. Візуалізація результатів прогнозування ---")
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data, label='Тренувальні дані (Actual Train)')
plt.plot(test_data.index, test_data, label='Тестові дані (Actual Test)', color='orange')
# Тепер `predictions` можна використовувати для графіка, бо це Series з правильним індексом
plt.plot(comparison_df.index, comparison_df['Predicted'], label='Прогноз ARIMA (Predicted)', color='green',
         linestyle='--')
plt.title(f'Прогноз цін акцій {ticker} за допомогою ARIMA({p},{d},{q})')
plt.xlabel('Дата')
plt.ylabel('Ціна закриття (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Прогнозування часових рядів завершено ---")
