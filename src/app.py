from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from src.data_fetch import create_sequences

app = FastAPI()

# Mount static files for CSS and JavaScript
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up template directory
templates = Jinja2Templates(directory="templates")
scaler = MinMaxScaler()


@app.get("/")
async def read_root (request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict (request: Request):
    try:
        data = await request.json()
        stock_name = data.get("stock_name")
        days = int(data.get("minutes"))
        interval = data.get("interval", "1m")
        
        model_path = f"../models/{stock_name}/{interval}_{stock_name}_best_model.h5"
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")
        
        period = "2y" if interval == "1h" else "max"
        df = yf.download(stock_name, period=period, interval=interval)
        
        df.index = pd.to_datetime(df.index)
        df = df[["Adj Close"]].dropna()
        scaled_data = scaler.fit_transform(df)
        
        x_seq, _ = create_sequences(scaled_data)
        last_seq = x_seq[-1].reshape(1, x_seq.shape[1], x_seq.shape[2])
        
        predictions = []
        for _ in range(days):
            pred = model.predict(last_seq)
            last_seq = np.append(last_seq[:, 1:, :], np.expand_dims(pred, axis=1), axis=1)
            predictions.append(pred[0, 0])
        
        predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        return JSONResponse(content={"predicted_price": predictions_rescaled.tolist()})
    
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
