# Forecasting Models for Time Series Analysis

Time Series Forecasting refers to predicting future values based on previously observed values (historical data). Forecasting is essential in industries like stock markets, weather prediction, and demand forecasting.

---

## 1. Key Libraries for Time Series Forecasting

1. **NumPy**: For numerical operations and data manipulation.
   ```bash
   pip install numpy
   ```
2. **pandas**: For time-series data preprocessing and analysis.
   ```bash
   pip install pandas
   ```
3. **Matplotlib & Seaborn**: For visualizing trends and patterns.
   ```bash
   pip install matplotlib seaborn
   ```
4. **statsmodels**: For implementing statistical models (like ARIMA).
   ```bash
   pip install statsmodels
   ```
5. **scikit-learn**: For ML-based forecasting techniques.
   ```bash
   pip install scikit-learn
   ```
6. **pmdarima**: For automating ARIMA model selection.
   ```bash
   pip install pmdarima
   ```
7. **Prophet (Meta/Open)**: A tool designed for accurate and fast forecasting.
   ```bash
   pip install prophet
   ```

---

## 2. Time Series Components

Time series data often contains four main components:
1. **Trend**: Long-term upward or downward movement.
2. **Seasonality**: Periodic fluctuations (e.g., weekly, yearly).
3. **Cyclic Variations**: Recurring fluctuations, not of a fixed period (e.g., economic cycles).
4. **Residuals**: Random noise or irregularities.

### Example of Decomposition
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Load Data
data = pd.read_csv("your_time_series.csv")  # Ensure you have a column with a datetime index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Step 2: Decompose the Time Series
result = seasonal_decompose(data['Value'], model='additive', period=12)  # Specify period if known
result.plot()
plt.show()
```

---

## 3. Forecasting Models

### 3.1 Naïve Forecasting
Assumes the next observation is equal to the last observed value. It serves as a baseline for evaluating complex models.
```python
# Naive forecasting
def naive_forecast(data):
    return data[-1]  # Forecast is the last observed value

data = [50, 52, 54, 56]
forecast = naive_forecast(data)
print("Naive Forecast:", forecast)
```

---

### 3.2 Moving Average (SMA/EMA)

1. **Simple Moving Average (SMA)**:
   Calculates the average of recent `k` observations.
   ```python
   import pandas as pd
   
   # Simulating Example Data
   data = pd.Series([120, 130, 140, 150, 160, 170, 180])
   window = 3  # Window size for moving average
   
   # Calculating Moving Average
   moving_avg = data.rolling(window).mean()
   print(moving_avg)
   ```

2. **Exponential Moving Average (EMA)**:
   Gives more weight to recent observations by using a smoothing factor.
   ```python
   alpha = 0.3  # Smoothing factor
   data = [120, 130, 140, 150, 160, 170]
   
   # Manual EMA Calculation
   ema = [data[0]]  # First value as initial EMA
   for i in range(1, len(data)):
       ema.append(alpha*data[i] + (1-alpha)*ema[-1])
   
   print("Exponential Moving Average:", ema)
   ```

---

### 3.3 ARIMA (Autoregressive Integrated Moving Average)

ARIMA models are suitable for stationary time series. The model is defined by parameters `(p, d, q)`:
1. **p**: Order of autoregression (AR).
2. **d**: Differencing to make the series stationary.
3. **q**: Order of the moving average (MA).

```python
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Step 1: Load Data (you need historical time series data)
data = [12, 15, 14, 18, 20, 23, 22, 25, 29, 32]

# Step 2: Fit ARIMA Model
model = ARIMA(data, order=(2, 1, 2))  # ARIMA(p,d,q)
fitted_model = model.fit()

# Step 3: Forecast
forecast = fitted_model.forecast(steps=5)  # Predict next 5 steps
print("Forecast:", forecast)

# Step 4: Plot Actual vs Forecasted
plt.plot(data, label='Actual Data')
plt.plot(range(len(data), len(data) + len(forecast)), forecast, label='Forecast', color='red')
plt.legend()
plt.show()
```

---

### 3.4 Auto ARIMA (Automated ARIMA Selection)

`pmdarima` automates ARIMA parameter selection.
```python
from pmdarima import auto_arima

# Load Example Observation Series
data = [12, 15, 14, 16, 20, 23, 22, 25, 29]  # Replace with your time series data

# Build Auto ARIMA Model
model = auto_arima(data, seasonal=False, suppress_warnings=True)  # For non-seasonal
forecast = model.predict(n_periods=5)  # Forecast next 5 periods

print("Auto ARIMA Forecast:", forecast)
```

---

### 3.5 Seasonal Decomposition of ARIMA (SARIMA)

SARIMA is an extension of ARIMA designed for **seasonal data**. It uses additional parameters `(P, D, Q, S)` for seasonality.

Example:
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Seasonal Order → (P, D, Q, S): S is the periodicity (e.g., 12 for monthly data)
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fit = model.fit()
forecast = fit.forecast(steps=12)  # Next 12 periods
print("SARIMA Forecast:", forecast)
```

---

### 3.6 Prophet (Facebook/Meta)

Prophet is widely used for both trend and seasonality forecasting.

```python
from prophet import Prophet
import pandas as pd

# Step 1: Prepare Data in Required Format
data = pd.DataFrame({
    'ds': pd.date_range(start='2021-01-01', periods=30, freq='D'),  # Date column
    'y': [120, 130, 140, 150, 160, 170] + list(range(175, 205))  # Observations
})

# Step 2: Fit Prophet Model
model = Prophet()
model.fit(data)

# Step 3: Create Future Dates and Forecast
future = model.make_future_dataframe(periods=10)  #10 extra days
forecast = model.predict(future)

# Step 4: Plot Forecast
model.plot(forecast)
plt.show()
```

---

### 3.7 Long Short-Term Memory (LSTM – Neural Networks)

LSTMs are recurrent neural networks (RNNs) designed for sequential data like time series.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Prepare Data
data = np.array([120, 130, 140, 150, 160, 170, 180])
sequence_length = 3  # Rolling window size
X, y = [], []

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X, y = np.array(X), np.array(y)

# Step 2: Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Step 3: Train the Model
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM
model.fit(X, y, epochs=200, verbose=0)

# Step 4: Forecast
new_data = np.array([150, 160, 170]).reshape((1, 3, 1))  # Next sequence
forecast = model.predict(new_data)
print("LSTM Forecast:", forecast)
```

---

### Practical Tips:
1. **Data Preprocessing**:
   - Check for missing values and handle them using interpolation or moving averages.
   - Perform stationarity checks (e.g., the Augmented Dickey-Fuller test).
   
2. **Feature Engineering**:
   - Add lag features, rolling means, or external covariates like holidays.

3. **Evaluation Metrics**:
   - **Mean Absolute Error (MAE)**:
     ```python
     from sklearn.metrics import mean_absolute_error
     print(mean_absolute_error(y_true, y_pred))
     ```
   - **Mean Squared Error (MSE)**
   - **R-squared (R²)**

---

This guide progresses from simple techniques like Naïve and Moving Averages to advanced time-series models like ARIMA, SARIMA, LSTMs, and Prophet, ensuring all aspects of forecasting are covered comprehensively!