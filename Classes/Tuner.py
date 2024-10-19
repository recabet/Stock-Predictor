from itertools import product
from typing import List, Tuple, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Backend.data_fetch import create_sequences, prepare_data
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import keras
import numpy as np


class Tuner:
    def __init__ (self,
                  epochs_list: List[int] = None,
                  batch_size_list: List[int] = None,
                  lstm_units_list: List[int] = None,
                  seq_length_list: List[int] = None,
                  num_layers_list: List[int] = None,
                  model_builder: Callable = None) -> None:
        
        if epochs_list is None:
            epochs_list = [5, 10, 15]
        if batch_size_list is None:
            batch_size_list = [16, 24, 32]
        if lstm_units_list is None:
            lstm_units_list = [32, 64, 96]
        if seq_length_list is None:
            seq_length_list = [20, 25, 30]
        if num_layers_list is None:
            num_layers_list = [1, 2, 3]
        
        self.epochs_list: List[int] = epochs_list
        self.batch_size_list: List[int] = batch_size_list
        self.lstm_units_list: List[int] = lstm_units_list
        self.seq_length_list: List[int] = seq_length_list
        self.num_layers_list: List[int] = num_layers_list
        self.hyperparameters: List[Tuple[int, int, int, int, int]] = list(
            product(self.epochs_list, self.batch_size_list, self.seq_length_list, self.lstm_units_list,
                    self.num_layers_list)
        )
        
        self.model_builder: Callable = model_builder if model_builder else self.__build_model
    
    @staticmethod
    def __build_model (lstm_units: int, seq_length: int, feature_dim: int, num_layers: int) -> keras.Model:
        model = keras.Sequential()
        model.add(
            keras.layers.LSTM(lstm_units, input_shape=(seq_length, feature_dim), return_sequences=(num_layers > 1))
        )
        for _ in range(1, num_layers):
            model.add(keras.layers.LSTM(lstm_units, return_sequences=(_ < num_layers - 1)))
        model.add(keras.layers.Dense(1))
        
        return model
    
    def tune (self, stock_symbol: str, interval: str, threshold: float = 5, metric: str = "mse",
              plot: bool = False, verbose: bool = False):
        best_model = None
        best_mse = float('inf')
        
        for params in self.hyperparameters:
            epochs, batch_size, seq_length, lstm_units, num_layers = params
            
            scaler = MinMaxScaler()
            train_data_scaled, test_data_scaled = prepare_data(stock_symbol, scaler, interval)
            X_train, y_train = create_sequences(train_data_scaled, seq_length=seq_length)
            X_test, y_test = create_sequences(test_data_scaled, seq_length=seq_length)
            
            model = self.model_builder(lstm_units, X_train.shape[1], X_train.shape[2], num_layers)
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=verbose)
            
            y_pred = model.predict(X_test)
            y_pred_reshaped = y_pred.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
            y_pred_rescaled = scaler.inverse_transform(y_pred_reshaped)
            y_test_rescaled = scaler.inverse_transform(y_test_reshaped)
            
            match metric:
                case "mse":
                    evaluation_metric = mean_squared_error(y_test_rescaled, y_pred_rescaled)
                case "mae":
                    evaluation_metric = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
                case "rmse":
                    evaluation_metric = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
                case _:
                    raise ValueError(f"Unsupported metric: {metric}")
            
            if verbose:
                print(
                    f"Model with: epochs={epochs}, batch_size={batch_size}, seq_length={seq_length}, "
                    f"lstm_units={lstm_units}, num_layers={num_layers}"
                )
                print(f"{metric.upper()}: {evaluation_metric:.4f}")
            
            if plot:
                plt.figure(figsize=(14, 7))
                plt.plot(y_test_rescaled, color="blue", label="Actual Price")
                plt.plot(y_pred_rescaled, color="red", label="Predicted Price")
                plt.title(f"Actual vs Predicted Stock Prices ({stock_symbol})")
                plt.xlabel("Time")
                plt.ylabel("Stock Price")
                plt.legend()
                plt.text(0.95, 0.75, f"{metric.upper()}: {evaluation_metric:.2f}",
                         transform=plt.gca().transAxes, fontsize=12, verticalalignment="top",
                         horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
                plt.show()
            
            if evaluation_metric < best_mse:
                best_mse = evaluation_metric
                best_model = model
            
            if best_mse < threshold:
                print(f"Early stopping, {metric.upper()} achieved: {best_mse:.4f}")
                break
        
        return best_model
