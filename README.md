Crypto Insight: AI-Powered Price Tracker

Overview
Crypto Insight is a web application that predicts cryptocurrency prices using machine learning. It fetches real-time data and applies a linear regression model to forecast short-term trends.

Features
Real-time Data Fetching: Pulls cryptocurrency prices from the CoinGecko API.
Machine Learning Predictions: Uses Linear Regression to predict future prices.
Performance Metrics: Provides MSE (Mean Squared Error) and R² (R-squared) for model accuracy.
Flexible API Endpoint: Supports multiple cryptocurrencies like Bitcoin and Ethereum.
Dashboard UI: Displays instructions for using the app.
How It Works
The application fetches the last 30 days' price data for a selected cryptocurrency.
Prepares data for training and testing a linear regression model.
Predicts future prices and evaluates model performance using MSE and R².
Returns the results in JSON format via an API endpoint.
Installation
1. Clone the Repository:

git clone <repo-url>
cd crypto-tracker-ai
2. Create a Virtual Environment:

python3.12 -m venv venv
source venv/bin/activate
3. Install Dependencies:

pip install -r requirements.txt
4. Run the Flask App:

python app.py
Usage
Open the app in your browser:
http://127.0.0.1:5000/
Access predictions:
http://127.0.0.1:5000/predict?crypto=bitcoin
Replace bitcoin with any supported cryptocurrency.

Why It's Good
Accurate Predictions: Uses proven ML algorithms to predict trends.
Easy to Use: RESTful API endpoints for seamless integration.
Customizable: Extendable to support additional features like graphs, alerts, and portfolio tracking.
Educational: Great for learning Flask, APIs, and ML modeling.
