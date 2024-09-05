from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load model and scaler
model = tf.keras.models.load_model('model.h5')
scaler = MinMaxScaler()


@app.route('/predict', methods=['POST'])
def predict ():
    try:
        data = request.json
        stock_name = data['stock_name']
        days = int(data['days'])
        
        # Fetch stock data
        df = yf.download(stock_name, period="max", interval="1d")
        df.index = pd.to_datetime(df.index)
        
        # Prepare data
        df = df[['Adj Close']].dropna()  # Ensure no missing values
        scaled_data = scaler.fit_transform(df)
        
        # Create sequences
        x_seq, _ = create_sequences(scaled_data)
        
        last_seq = x_seq[-1].reshape(1, x_seq.shape[1], x_seq.shape[2])
        
        predictions = []
        
        # Predict for the specified number of days
        for _ in range(days):
            pred = model.predict(last_seq)
            
            # Update the sequence: remove the oldest and add the newest prediction
            # Reshape the prediction to match the shape (samples, timesteps, features)
            last_seq = np.append(last_seq[:, 1:, :], np.expand_dims(pred, axis=1), axis=1)
            
            predictions.append(pred[0, 0])
        
        # Rescale the predictions back to original scale
        predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        return jsonify({'predicted_price': predictions_rescaled.tolist()})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


def create_sequences (datas, seq_length=60):
    x, y = [], []
    for i in range(len(datas) - seq_length):
        x.append(datas[i:i + seq_length])
        y.append(datas[i + seq_length])
    return np.array(x), np.array(y)


@app.after_request
def after_request (response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


if __name__ == '__main__':
    app.run(debug=True)
