import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random

SEED=6
np.random.seed(SEED)
random.seed(SEED)

scaler = MinMaxScaler()

def prepare_data(stock_name:str,scaler_,interval:str="1h"):
    match interval:
        case "1h":
            data = yf.download(f"{stock_name}", period="2y", interval=interval)
        case "1m":
            data = yf.download(f"{stock_name}", period="max", interval=interval)
        case "1d":
            data = yf.download(f"{stock_name}", period="max", interval=interval)
        case _:
            raise ValueError(f"Invalid interval {interval}")
        

    data.index = pd.to_datetime(data.index)
    print(data.index.max())
    data["Daily_Ret"] = data["Adj Close"].pct_change()
    data.dropna(inplace=True)
    train_data, test_data = train_test_split(data[["Adj Close"]], test_size=0.2, shuffle=False,random_state=SEED)
    train_data_sc = scaler_.fit_transform(train_data[["Adj Close"]])
    test_data_sc = scaler_.transform(test_data[["Adj Close"]])
    return train_data_sc, test_data_sc


def create_sequences (datas, seq_length=23):
    x, y = [], []
    for i in range(len(datas) - seq_length):
        x.append(datas[i:i + seq_length])
        y.append(datas[i + seq_length])
    return np.array(x), np.array(y)





