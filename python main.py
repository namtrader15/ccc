from binance.client import Client
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os
from datetime import datetime

def get_realtime_klines(symbol, interval, lookback, client):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].astype(float)
    return data

def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def analyze_trend(interval, client):
    symbol = 'BTCUSDT'
    lookback = 1000
    data = get_realtime_klines(symbol, interval, lookback, client)
    
    rsi = calculate_rsi(data)
    macd, signal_line = calculate_macd(data)
    
    data['rsi'] = rsi
    data['macd'] = macd
    data['signal_line'] = signal_line
    
    data = data.dropna()
    
    features = data[['rsi', 'macd', 'signal_line']]
    target = (data['close'].shift(-1) > data['close']).astype(int)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target.dropna(), test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    latest_features = features_scaled[-1].reshape(1, -1)
    prediction_prob = model.predict_proba(latest_features)[0]
    threshold = 0.45
    
    if prediction_prob[1] > 1 - threshold:
        return "Tăng"
    elif prediction_prob[1] < threshold:
        return "Giảm"
    else:
        return "Không rõ ràng"

def main():
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    client = Client(api_key, api_secret, tld='com', testnet=False)
    
    trend_1m = analyze_trend(Client.KLINE_INTERVAL_1MINUTE, client)
    trend_15m = analyze_trend(Client.KLINE_INTERVAL_15MINUTE, client)
    trend_4h = analyze_trend(Client.KLINE_INTERVAL_4HOUR, client)
    
    print(f"Xu hướng 1 phút: {trend_1m}")
    print(f"Xu hướng 15 phút: {trend_15m}")
    print(f"Xu hướng 4 giờ: {trend_4h}")

if __name__ == "__main__":
    main()
