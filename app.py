import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",  # Laravel's frontend URL
    "http://127.0.0.1:8000",  # Localhost for FastAPI
    "http://localhost:8000",  # Adjust if your frontend is hosted elsewhere
    "https://a8f2-139-135-241-49.ngrok-free.app",
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

    # Add seasonal and trend features
    data_pivot['month'] = data_pivot.index.month
    data_pivot['day_of_week'] = data_pivot.index.dayofweek

    # Create lag features for each waste type
    for col in ['Biodegradable', 'Residual', 'Recyclable']:
        data_pivot[f'{col}_lag1'] = data_pivot[col].shift(30)
        data_pivot[f'{col}_rolling_mean'] = data_pivot[col].rolling(window=3).mean()

    data_pivot.dropna(inplace=True)

    # Define features (X) and target (y)
    feature_cols = [
        'Biodegradable_lag1', 'Residual_lag1', 'Recyclable_lag1',
        'Biodegradable_rolling_mean', 'Residual_rolling_mean', 'Recyclable_rolling_mean',
        'month', 'day_of_week'
    ]
    X = data_pivot[feature_cols]
    y = data_pivot[['Biodegradable', 'Residual', 'Recyclable']]

    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    tscv = TimeSeriesSplit(n_splits=5)

    # Model selection and hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)

    # Best model after hyperparameter tuning
    model = grid_search.best_estimator_

    # Predict the target for the entire dataset
    y_pred = model.predict(X_scaled)

    # Performance metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    smape = symmetric_mape(y, y_pred)

    performance_metrics = {
        "mse": mse,
        "mae": mae,
        "smape": smape,
        "r2": r2
    }

    # Predict next month
    last_data_point = X_scaled[-1].reshape(1, -1)  # Use the last data point for prediction
    next_month_prediction = model.predict(last_data_point)

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

# @app.get("/byproducts-prediction", response_class=PlainTextResponse)
# async def byproducts_prediction():
#     try:
#         response = requests.get("http://localhost/gtms/public/api/byproducts-data")
#         response.raise_for_status()
#     except requests.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch data from Laravel API: {e}")

#     # Verify JSON response
#     try:
#         data_json = response.json()
#     except ValueError:
#         raise HTTPException(status_code=500, detail="Invalid JSON format from Laravel API")

#     # Convert JSON data to DataFrame to ensure it is processed correctly
#     data = pd.DataFrame(data_json)
#     if data.empty:
#         raise HTTPException(status_code=500, detail="No data received from Laravel API")

#     return "Data fetched successfully!"