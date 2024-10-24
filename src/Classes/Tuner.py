import keras
from keras_tuner import RandomSearch, HyperParameters
from src.data_fetch import create_sequences, prepare_data, scaler
from src.Classes.ConvertToH5Callback import _ConvertToH5Callback
from matplotlib import pyplot as plt
from typing import Optional, List, Dict


class TuningException(Exception):
    """Custom exception for Tuning errors."""
    pass


class Tuner:
    def __init__ (self,
                  fmt: str,
                  max_trials: int = 10,
                  executions_per_trial: int = 3,
                  directory: str = "tuner_dir") -> None:
        
        self.max_trials: int = max_trials
        self.executions_per_trial: int = executions_per_trial
        self.directory: str = directory
        self.project_name: str = ""
        self.tuner: Optional[RandomSearch] = None
        self.fmt: str = fmt
    
    @staticmethod
    def __validate_dropout_params (dropout: bool, dropout_deg: float) -> None:
        if dropout and not (0 < dropout_deg < 1):
            raise TuningException(f"Invalid dropout_deg value: {dropout_deg}. It should be between 0 and 1.")
    
    @staticmethod
    def __build_model (hp: HyperParameters, dropout: bool = False, dropout_deg: float = 0.0) -> keras.Model:
        try:
            Tuner.__validate_dropout_params(dropout, dropout_deg)
            
            lstm_units = hp.Int("lstm_units", min_value=32, max_value=128, step=32)
            num_layers = hp.Int("num_layers", min_value=1, max_value=3)
            seq_length = hp.Int("seq_length", min_value=20, max_value=40, step=5)
            learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
            
            model = keras.Sequential()
            model.add(keras.layers.LSTM(lstm_units, input_shape=(seq_length, 1), return_sequences=(num_layers > 1)))
            for i in range(1, num_layers):
                model.add(keras.layers.LSTM(lstm_units, return_sequences=(i < num_layers - 1)))
                if dropout:
                    model.add(keras.layers.Dropout(dropout_deg))
            model.add(keras.layers.Dense(1))
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error")
            return model
        except Exception as e:
            raise TuningException(f"Error building model: {str(e)}")
    
    @staticmethod
    def __plot (model, X, Y, stock_symbol: str):
        y_pred = model.predict(X)
        y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
        Y_rescaled = scaler.inverse_transform(Y.reshape(-1, 1))
        
        plt.figure(figsize=(14, 7))
        plt.plot(Y_rescaled, color="blue", label="Actual Price")
        plt.plot(y_pred_rescaled, color="red", label="Predicted Price")
        plt.title(f"Actual vs Predicted Stock Prices ({stock_symbol})")
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()
    
    def tune (self,
              stock_symbol: str,
              interval: str,
              epochs: int = 10,
              batch_size: int = 32,
              metric: str = "val_loss",
              plot: bool = False,
              verbose: bool = True) -> Optional[keras.Model]:
        try:
            self.project_name = f"{stock_symbol}_{interval}_tuning_hist"
            self.tuner = RandomSearch(
                self.__build_model,
                objective=metric,
                max_trials=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                directory=self.directory,
                project_name=self.project_name
            )
            
            train_data_scaled, test_data_scaled = prepare_data(stock_symbol, interval)
            if train_data_scaled.size == 0 or test_data_scaled.size == 0:
                raise TuningException("Training or testing data is empty.")
            
            X_train, y_train = create_sequences(train_data_scaled)
            X_val, y_val = create_sequences(test_data_scaled)
            
            keras_filepath = f"/home/recabet/Coding/Stock-Predictor/models/{interval}_{stock_symbol}_best_model.keras"
            h5_filepath = f"/home/recabet/Coding/Stock-Predictor/models/{interval}_{stock_symbol}_best_model.h5"
            
            callbacks = [
                keras.callbacks.EarlyStopping(monitor=metric,
                                              patience=5,
                                              restore_best_weights=True,
                                              verbose=1),
                
                keras.callbacks.ReduceLROnPlateau(monitor=metric,
                                                  factor=0.5,
                                                  patience=3,
                                                  verbose=1,
                                                  min_lr=1e-6),
                
                keras.callbacks.ModelCheckpoint(filepath=keras_filepath,
                                                monitor=metric,
                                                save_best_only=True,
                                                verbose=1),
                
                _ConvertToH5Callback(keras_filepath=keras_filepath, h5_filepath=h5_filepath)
            ]
            
            self.tuner.search(X_train,
                              y_train,
                              epochs=epochs,
                              validation_data=(X_val, y_val),
                              batch_size=batch_size,
                              callbacks=callbacks,
                              verbose=verbose)
            
            best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
            print(f"Best LSTM units: {best_hps.get('lstm_units')}")
            print(f"Best number of layers: {best_hps.get('num_layers')}")
            print(f"Best sequence length: {best_hps.get('seq_length')}")
            print(f"Best learning rate: {best_hps.get('learning_rate')}")
            
            best_model = self.tuner.hypermodel.build(best_hps)
            best_model.fit(X_train,
                           y_train,
                           epochs=epochs,
                           validation_data=(X_val, y_val),
                           batch_size=batch_size,
                           callbacks=callbacks,
                           verbose=verbose)
            
            if plot:
                self.__plot(best_model, X_val, y_val, stock_symbol)
            
            return best_model
        except Exception as e:
            raise TuningException(f"Error during tuning: {str(e)}")
