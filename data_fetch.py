import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = yf.download("AAPL", period="max", interval="1d")

data.index = pd.to_datetime(data.index)
print(data.index.max())
data["Daily_Ret"] = data["Adj Close"].pct_change()
data.dropna(inplace=True)

train_data, test_data = train_test_split(data[["Adj Close"]], test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data[["Adj Close"]])
test_data_scaled = scaler.transform(test_data[["Adj Close"]])


def create_sequences (datas, seq_length=15):
    x, y = [], []
    for i in range(len(datas) - seq_length):
        x.append(datas[i:i + seq_length])
        y.append(datas[i + seq_length])
    return np.array(x), np.array(y)


X_train, y_train = create_sequences(train_data_scaled)
X_test, y_test = create_sequences(test_data_scaled)

sequential_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(1)
])

sequential_model.compile(optimizer="adam", loss="mean_squared_error")
sequential_model.fit(X_train, y_train, batch_size=32, epochs=25, validation_split=0.1)

y_pred = sequential_model.predict(X_test)

y_pred_reshaped = y_pred.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)
y_pred_rescaled = scaler.inverse_transform(y_pred_reshaped)
y_test_rescaled = scaler.inverse_transform(y_test_reshaped)

mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

sequential_model.save("model.h5")
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, color="blue", label="Actual Price")
plt.plot(y_pred_rescaled, color="red", label="Predicted Price")
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

last_sequence = train_data_scaled[-60:]

future_predictions = []

for _ in range(10):
    prediction_input = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    predicted_price = sequential_model.predict(prediction_input)
    
    future_predictions.append(predicted_price[0, 0])
    
    last_sequence = np.append(last_sequence[1:], predicted_price, axis=0)

future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

print("Future Predictions (next 10 days):")
print(future_predictions_rescaled)
