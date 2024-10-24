import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.SEEDS import set_seed
from typing import Tuple, List

SEED: int = 6
set_seed(SEED)

scaler = MinMaxScaler()


def scale_data (train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    global scaler
    train_scaled: np.ndarray = scaler.fit_transform(train_data)
    test_scaled: np.ndarray = scaler.transform(test_data)
    return train_scaled, test_scaled


def prepare_data (stock_name: str, interval: str = "1h") -> Tuple[np.ndarray, np.ndarray]:
    try:
        match interval:
            case "1h":
                data = yf.download(stock_name, period="2y", interval=interval)
            case "1m":
                data = yf.download(stock_name, period="max", interval=interval)
            case _:
                raise ValueError(f"Invalid interval {interval}")
        
        data.index = pd.to_datetime(data.index)
        data.dropna(inplace=True)
        train_data, test_data = train_test_split(data[["Adj Close"]],
                                                 test_size=0.2,
                                                 shuffle=False,
                                                 random_state=SEED)
        
        train_data_sc, test_data_sc = scale_data(train_data[["Adj Close"]], test_data[["Adj Close"]])
        
        return train_data_sc, test_data_sc
    
    except Exception as e:
        raise RuntimeError(f"Error preparing data: {e}")


def create_sequences (datas: np.ndarray, seq_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    x: List = []
    y: List = []
    for i in range(len(datas) - seq_length):
        x.append(datas[i:i + seq_length])
        y.append(datas[i + seq_length])
    return np.array(x), np.array(y)
