import os
import keras
from src.Classes.Tuner import Tuner
from typing import List, Dict


class ModelCreator(Tuner):
    def __init__ (self,
                  stock_symbol: str,
                  fmt: str,
                  intervals: List[str],
                  max_trials: int = 10,
                  executions_per_trial: int = 3,
                  directory: str = "tuner_dir",
                  ) -> None:
        
        super().__init__(fmt,
                         max_trials,
                         executions_per_trial,
                         directory)
        
        self.stock_symbol = stock_symbol
        self.intervals = intervals
        self.models = None
    
    def train_tune (self,
                    epochs: int = 10,
                    batch_size: int = 32,
                    metric: str = "val_loss",
                    plot: bool = False,
                    verbose: bool = True) -> Dict[str, keras.Model]:
        
        models: Dict[str, keras.Model] = {}
        for interval in self.intervals:
            model_name:str = f"{interval}_{self.stock_symbol}_model"
            print(f"Tuning for interval: {interval}")
            
            models[model_name] = self.tune(self.stock_symbol,
                                           interval,
                                           epochs,
                                           batch_size,
                                           metric,
                                           plot,
                                           verbose)
        
        self.models = models
        return models
    
    def save_models (self, directory: str) -> None:
        if not self.models:
            raise ValueError("No models to save. Please run `train_tune()` first.")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for model_name, model in self.models.items():
            model.save(os.path.join(directory, f"{model_name}{self.fmt}"))
            print(f"Model {model_name} saved.")
