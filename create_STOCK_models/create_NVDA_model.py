import numpy as np
import random
import tensorflow as tf
from Classes.ModelCreator import ModelCreator

SEED=6

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

interval = "1h"

creator=ModelCreator(
                     epochs_list=[5,10,20, 30, 35],
                     intervals=["1m","1h"],
                     batch_size_list=[8,16,24,32],
                     lstm_units_list=[64,88],
                     num_layers_list=[1,2],
                     seq_length_list=[1,2,5,6,15],
                     stock_symbol="NVDA"
                    )

models=creator.train_tune(thresholds={
    "1m":0.01,
    "1h":3,
},plot=True,verbose=True)

creator.save_models("models/NVDA")