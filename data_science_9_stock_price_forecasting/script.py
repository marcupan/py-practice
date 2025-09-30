import warnings
warnings.filterwarnings("ignore")

# Базові
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Дані
import yfinance as yf

# Статистика / моделі
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Метрики
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Бібліотеки імпортовано.")

# ---------------------------
# 1) Завантаження даних
# ---------------------------
ticker = 'AAPL'
start_date = '2018-01-01'
end_date = '2024-12-31'  # можна змінити на актуальну дату

print(f"\n--- 1. Завантаження даних для {ticker} з {start_date} по {end_date} ---")
try:
    # auto_adjust=True — прибирає спліти/дивіденди, зручно для моделювання
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("Завантажено порожній DataFrame. Перевірте тікер або період.")
    # Залишимо тільки ціну закриття, приберемо NaN
    ts_data = df['Close'].dropna().copy()
    # Переконаємось, що індекс — DatetimeIndex, без часової зони
    ts_data.index = pd.to_datetime(ts_data.index).tz_localize(None)
    print(f"Дані завантажено. Розмір: {ts_data.shape[0]} спостережень.")
except Exception as e:
    print(f"Помилка завантаження даних: {e}")
    raise SystemExit(1)

# ---------------------------
# 2) EDA + перевірка стаціонарності
# ---------------------------
print("\n--- 2. EDA та Перевірка на стаціонарність ---")
plt.figure(figsize=(14, 7))
ts_data.plot()
plt.title(f'Ціна закриття акцій {ticker}')
plt.xlabel('Дата')
plt.ylabel('Ціна закриття (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()

def check_stationarity(timeseries: pd.Series) -> bool:
    series = pd.Series(timeseries).dropna()
    print('\nРезультати тесту Дікі-Фуллера:')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    if dftest[1] <= 0.05:
        print("=> Результат: Ряд стаціонарний (відхиляємо H0)")
        return True
    else:
        print("=> Результат: Ряд НЕ стаціонарний (не відхиляємо H0)")
        return False

_ = check_stationarity(ts_data)

# ---------------------------
# 3) Диференціювання (до стаціонарності)
# ---------------------------
print("\n--- 3. Передобробка: Диференціювання ---")
ts_diff = ts_data.diff().dropna()

plt.figure(figsize=(14, 7))
ts_diff.plot()
plt.title(f'Різниця цін закриття {ticker} (1 порядок)')
plt.xlabel('Дата')
plt.ylabel('Δ Ціна')
plt.grid(True)
plt.tight_layout()
plt.show()

_ = check_stationarity(ts_diff)
d = 1  # порядок диференціювання

# ---------------------------
# 4) ACF/PACF → вибір p, q (фіксовані, але малюємо для огляду)
# ---------------------------
print("\n--- 4. Аналіз ACF та PACF для визначення p та q ---")
max_lags = max(1, min(40, len(ts_diff) - 1))  # обережно з короткими рядами
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(ts_diff, ax=axes[0], lags=max_lags)
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(ts_diff, ax=axes[1], lags=max_lags, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Спрощено: лишаємо як у завданні
p = 5
q = 5
print(f"Обрані параметри ARIMA: p={p}, d={d}, q={q}")

# ---------------------------
# 5) Розділення Train/Test
# ---------------------------
split_idx = int(len(ts_data) * 0.8)
train_data = ts_data.iloc[:split_idx]
test_data  = ts_data.iloc[split_idx:]

print("\n--- 5. Розділення даних ---")
print(f"Розмір train: {len(train_data)}")
print(f"Розмір test : {len(test_data)}")
print(f"Test від: {test_data.index.min().date()} до {test_data.index.max().date()}")

# ---------------------------
# 6) Навчання ARIMA
# ---------------------------
print("\n--- 6. Побудова та Тренування моделі ARIMA ---")
model = ARIMA(train_data, order=(p, d, q))
model_fit = model.fit()
print("\nЗведення моделі ARIMA:")
print(model_fit.summary())

# ---------------------------
# 7) Прогноз + оцінка
# ---------------------------
print("\n--- 7. Прогнозування на тесті та Оцінка ---")
n_forecast_steps = len(test_data)

# Зручно взяти get_forecast, щоб отримати інтервали і вирівняти індекс
fc = model_fit.get_forecast(steps=n_forecast_steps)
predictions = pd.Series(fc.predicted_mean.values, index=test_data.index, name='Predicted')

comparison_df = pd.DataFrame(
    {'Actual': test_data.values, 'Predicted': predictions.values},
    index=test_data.index
)
print("\nПорівняння перших 5 значень:")
print(comparison_df.head())

mae = mean_absolute_error(test_data, predictions)
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

mape = mean_absolute_percentage_error(test_data, predictions)

print(f"\nОцінка ARIMA({p},{d},{q}) на тесті:")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# ---------------------------
# 8) Візуалізація
# ---------------------------
print("\n--- 8. Візуалізація результатів прогнозування ---")
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data, label='Train')
plt.plot(test_data.index, test_data, label='Test', color='orange')
plt.plot(predictions.index, predictions, label='ARIMA forecast', color='green', linestyle='--')
plt.title(f'{ticker}: ARIMA({p},{d},{q}) прогноз')
plt.xlabel('Дата')
plt.ylabel('Ціна закриття (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Прогнозування часових рядів завершено ---")
