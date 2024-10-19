import os
from Classes.Tuner import Tuner
from typing import List,Dict


class ModelCreator(Tuner):
    def __init__ (self,
                  stock_symbol: str,
                  intervals: List[str],
                  epochs_list=None,
                  batch_size_list=None,
                  lstm_units_list=None,
                  seq_length_list=None,
                  num_layers_list=None):
        
        super().__init__(epochs_list,
                         batch_size_list,
                         lstm_units_list,
                         seq_length_list,
                         num_layers_list)
        
        self.stock_symbol: str = stock_symbol
        self.intervals: List[str] = intervals
        self.models = None
    
    def train_tune (self,
                    thresholds: Dict[str,float],
                    metric: str = "mse",
                    plot: bool = False,
                    verbose: bool = False):
        
        models = {}
        for interval in self.intervals:
            if interval not in thresholds:
                raise ValueError(f"Threshold for interval '{interval}' is not provided.")
            model_name = f"{interval}_{self.stock_symbol}_model"
            models[model_name] = self.tune(self.stock_symbol, interval, thresholds[interval], metric, plot, verbose)
        
        self.models = models
        return models
    
    def save_models (self, directory: str):
        if not self.models:
            raise ValueError("No models to save. Please run `train_tune()` first.")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for key, model in self.models.items():
            if model:
                model.save(os.path.join(directory, f"{key}.h5"))
            else:
                print(f"Warning: Model {key} has not been trained and will not be saved.")
