import pandas as pd
import requests
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from dateutil.relativedelta import relativedelta
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",  # Laravel's frontend URL
    "http://127.0.0.1:8000",  # Localhost for FastAPI
    "http://localhost:8000",  # Adjust if your frontend is hosted elsewhere
]

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    biodegradable: float
    residual: float
    recyclable: float
    start_date: str
    end_date: str
    performance_metrics: dict

def symmetric_mape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) to handle zero values.
    Returns the mean sMAPE score as a percentage.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)) * 100

@app.get("/predict-next-month", response_model=PredictionResponse)
async def predict_next_month():
    # Fetch data from Laravel API
    try:
        response = requests.get("http://localhost/gtms/public/api/waste-data")
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from Laravel API: {e}")

    # Verify JSON response
    try:
        data_json = response.json()
    except ValueError:
        raise HTTPException(status_code=500, detail="Invalid JSON format from Laravel API")

    # Convert JSON data to DataFrame
    data = pd.DataFrame(data_json)
    if 'collection_date' not in data.columns or 'waste_type' not in data.columns or 'metrics' not in data.columns:
        raise HTTPException(status_code=500, detail="Unexpected data format from Laravel API")

    # Convert 'collection_date' to datetime and 'metrics' to numeric
    data['date'] = pd.to_datetime(data['collection_date'], format='%d/%m/%Y')
    data['metrics'] = pd.to_numeric(data['metrics'], errors='coerce').fillna(0)

    # Pivot table to separate waste types
    data_pivot = data.pivot_table(index='date', columns='waste_type', values='metrics', fill_value=0)
    
    # Create lag features for each waste type
    data_pivot['biodegradable_lag1'] = data_pivot['Biodegradable'].shift(30, fill_value=0)
    data_pivot['residual_lag1'] = data_pivot['Residual'].shift(30, fill_value=0)
    data_pivot['recyclable_lag1'] = data_pivot['Recyclable'].shift(30, fill_value=0)
    
    # Create rolling mean features for trend (optional improvement)
    data_pivot['biodegradable_rolling_mean'] = data_pivot['Biodegradable'].rolling(window=3).mean()
    data_pivot['residual_rolling_mean'] = data_pivot['Residual'].rolling(window=3).mean()
    data_pivot['recyclable_rolling_mean'] = data_pivot['Recyclable'].rolling(window=3).mean()
    
    data_pivot.dropna(inplace=True)

    # Define features (X) and target (y)
    X = data_pivot[['biodegradable_lag1', 'residual_lag1', 'recyclable_lag1', 
                    'biodegradable_rolling_mean', 'residual_rolling_mean', 'recyclable_rolling_mean']]
    y = data_pivot[['Biodegradable', 'Residual', 'Recyclable']]

    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Use GridSearchCV to fine-tune the model
    ridge = Ridge()
    lasso = Lasso()

    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100],  # More granular search for Ridge and Lasso
        'fit_intercept': [True, False]     # Test both with and without intercept
    }

    # Fine-tune the model using Ridge or Lasso
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Best model after hyperparameter tuning
    model = grid_search.best_estimator_

    # Predict the target for both train and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Performance metrics for testing data
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    # Calculate sMAPE for the test set
    smape = symmetric_mape(y_test, y_test_pred)

    # Prepare performance metrics
    performance_metrics = {
        "mse": mse,
        "mae": mae,
        "mape": smape,  # Using sMAPE instead of MAPE
        "accuracy": r2  # Using R2 as accuracy
    }

    # Predict next month
    last_month_data = X_scaled[-1:].reshape(1, -1)  # Use the last data point for prediction
    next_month_prediction = model.predict(last_month_data)

    # Get next month’s date range
    current_date = datetime.now()
    next_month_start = (current_date + relativedelta(months=1)).replace(day=1)
    next_month_end = (next_month_start + relativedelta(months=1)) - pd.Timedelta(days=1)

    # Prepare and return response
    return PredictionResponse(
        biodegradable=next_month_prediction[0][0],
        residual=next_month_prediction[0][1],
        recyclable=next_month_prediction[0][2],
        start_date=str(next_month_start.date()),
        end_date=str(next_month_end.date()),
        performance_metrics=performance_metrics
    )
