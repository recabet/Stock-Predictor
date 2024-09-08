from data_fetch import create_sequences
from data_fetch import prepare_data
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from itertools import product


np.random.seed(6)
random.seed(6)


epochs_list = [30, 35, 40, 50]
batch_size_list = [16, 24, 32]
seq_length_list = [20, 23, 25, 30]
lstm_units_list = [64, 88, 100, 128]


hyperparameters = list(product(epochs_list, batch_size_list, seq_length_list, lstm_units_list))

best_mse = float("inf")
best_params = None

np.random.seed(6)
random.seed(6)

for params in hyperparameters:
    epochs, batch_size, seq_length, lstm_units = params
    print(f"Testing with: epochs={epochs}, batch_size={batch_size}, seq_length={seq_length}, lstm_units={lstm_units}")
    
    scaler = MinMaxScaler()
    train_data_scaled, test_data_scaled = prepare_data("NVDA", scaler)
    
    X_train, y_train = create_sequences(train_data_scaled, seq_length=seq_length)
    X_test, y_test = create_sequences(test_data_scaled, seq_length=seq_length)
    
    # Define the model with the current hyperparameters
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    
    sequential_model.compile(optimizer="adam", loss="mean_squared_error")
    
    # Train the model
    sequential_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=True)
    
    y_pred = sequential_model.predict(X_test)
    
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    y_pred_rescaled = scaler.inverse_transform(y_pred_reshaped)
    y_test_rescaled = scaler.inverse_transform(y_test_reshaped)
    
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    
    print(f"Mean Squared Error: {mse}, Mean Absolute Error: {mae}")
    
    if mse < best_mse:
        best_mse = mse
        best_model=sequential_model
        if best_mse<3:
            break


    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_rescaled, color="blue", label="Actual Price")
    plt.plot(y_pred_rescaled, color="red", label="Predicted Price")
    plt.title("Actual vs Predicted Stock Prices")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.text(0.95, 0.75, f"MSE: {mse:.2f}\nMAE: {mae:.2f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment="top",
             horizontalalignment="right", bbox=dict(boxstyle="round",
                                                    facecolor="white", alpha=0.5))
    plt.show()

best_model.save("1hr_NVDA_model.h5")

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
