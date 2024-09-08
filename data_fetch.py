import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


scaler = MinMaxScaler()

def prepare_data(stock_name:str,scaler_):
    data = yf.download(f"{stock_name}", period="2y", interval="1h")
    data.index = pd.to_datetime(data.index)
    print(data.index.max())
    data["Daily_Ret"] = data["Adj Close"].pct_change()
    data.dropna(inplace=True)
    train_data, test_data = train_test_split(data[["Adj Close"]], test_size=0.2, shuffle=False)
    train_data_sc = scaler_.fit_transform(train_data[["Adj Close"]])
    test_data_sc = scaler_.transform(test_data[["Adj Close"]])
    return train_data_sc, test_data_sc


def create_sequences (datas, seq_length=23):
    x, y = [], []
    for i in range(len(datas) - seq_length):
        x.append(datas[i:i + seq_length])
        y.append(datas[i + seq_length])
    return np.array(x), np.array(y)





