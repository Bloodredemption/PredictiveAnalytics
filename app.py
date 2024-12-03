import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from dateutil.relativedelta import relativedelta
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    mean_accuracy: float  # Mean accuracy of the model

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
    data_pivot.dropna(inplace=True)

    # Define features (X) and target (y)
    X = data_pivot[['biodegradable_lag1', 'residual_lag1', 'recyclable_lag1']]
    y = data_pivot[['Biodegradable', 'Residual', 'Recyclable']]

    # Initialize variables for mean accuracy and train/test sets
    mean_accuracy = None
    X_train, X_test, y_train, y_test = None, None, None, None

    if len(X) < 2:
        # Not enough data to split; train with all data
        model = LinearRegression()
        model.fit(X, y)
    else:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Calculate mean accuracy (R² score)
        y_pred = model.predict(X_test)
        mean_accuracy = r2_score(y_test, y_pred)

    # Predict next month
    last_month_data = X.tail(1)
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
        mean_accuracy=mean_accuracy if mean_accuracy is not None else 0.0  # Return 0.0 if not calculated
    )

# uvicorn app:app --reload