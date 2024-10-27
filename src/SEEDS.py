import numpy as np
import random
import tensorflow.random

def set_seed(SEED:int):
    np.random.seed(SEED)
    random.seed(SEED)
    tensorflow.random.set_seed(SEED)