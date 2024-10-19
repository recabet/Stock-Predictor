import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import tensorflow as tf
from Classes.Tuner import Tuner
from Backend.data_fetch import prepare_data
from typing import Callable

np.random.seed(6)
random.seed(6)
tf.random.set_seed(6)

interval = "1h"
def model_builder(lstm_units: int, seq_length: int=20, feature_dim: int=1, num_layers: int=1)->Callable:
    def _model_builder (_lstm_units: int, _seq_length: int, _feature_dim: int, _num_layers: int) -> tf.keras.Model:
        model = keras.Sequential()
        model.add(
            keras.layers.LSTM(lstm_units, input_shape=(seq_length, feature_dim), return_sequences=(num_layers > 1)))
        for _ in range(1, num_layers):
            model.add(keras.layers.LSTM(lstm_units, return_sequences=(_ < num_layers - 1)))
        model.add(keras.layers.Dense(1))
        return model
    return _model_builder(lstm_units, seq_length, feature_dim, num_layers)

tuner = Tuner(
    epochs_list=[20, 30, 35],
    batch_size_list=[24, 32],
    lstm_units_list=[64, 88],
    seq_length_list=[23],
    num_layers_list=[1, 2],
    model_builder=model_builder
)

best_model = tuner.tune(stock_symbol="AAPL", interval=interval, metric="mse",threshold=3, plot=True, verbose=True)
best_model.save(f"{interval}_AAPL_model.h5")
scaler = MinMaxScaler()
train_data_scaled, _ = prepare_data("AAPL", scaler, interval)