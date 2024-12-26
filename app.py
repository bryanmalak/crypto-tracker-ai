## Crypto Insight: AI-Powered Price Tracker

# Import required libraries
import os
import requests
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Flask app
app = Flask(__name__)

# Function to fetch real-time cryptocurrency data
def fetch_crypto_data(crypto='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '30',
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = [price[1] for price in data['prices']]
    return prices

# Prepare data for ML model
def prepare_data(prices):
    data = pd.DataFrame(prices, columns=['Price'])
    data['Target'] = data['Price'].shift(-1)
    data.dropna(inplace=True)
    X = data[['Price']]
    y = data['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML Model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Predict and Evaluate
# Update evaluate_model function
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Ensure the 'static' directory exists
    os.makedirs('static', exist_ok=True)
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label='Actual Prices')
    plt.plot(range(len(predictions)), predictions, label='Predicted Prices')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/prediction_plot.png')
    plt.close()
    
    return predictions, mse, r2


# Flask route to display dashboard
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to fetch and predict
@app.route('/predict', methods=['GET'])
def predict():
    crypto = request.args.get('crypto', 'bitcoin')
    prices = fetch_crypto_data(crypto)
    X_train, X_test, y_train, y_test = prepare_data(prices)
    model = train_model(X_train, y_train)
    predictions, mse, r2 = evaluate_model(model, X_test, y_test)

    # Render results in template
    return render_template('results.html',
                           mse=mse,
                           r2=r2,
                           predictions=predictions.tolist(),
                           chart_url='/static/prediction_plot.png')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, threaded=True)

