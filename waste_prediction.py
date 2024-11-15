import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta
from datetime import datetime

# Load data from CSV (replace 'waste_data.csv' with your actual file path)
data = pd.read_csv('waste_data.csv')

# Convert date column to datetime format
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

# Pivot table to separate waste types for each date
data_pivot = data.pivot_table(index='date', columns='waste_type', values='metrics', fill_value=0)

# Create lag features (previous month data as features for the model)
data_pivot['biodegradable_lag1'] = data_pivot['biodegradable'].shift(30, fill_value=0)
data_pivot['residual_lag1'] = data_pivot['residual'].shift(30, fill_value=0)

# Drop rows with no lagged values
data_pivot.dropna(inplace=True)

# Define features (X) and target (y)
X = data_pivot[['biodegradable_lag1', 'residual_lag1']]
y = data_pivot[['biodegradable', 'residual']]

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Predict waste metrics for the next month based on the current date
current_date = datetime.now()
next_month_start = (current_date + relativedelta(months=1)).replace(day=1)
next_month_end = (next_month_start + relativedelta(months=1)) - pd.Timedelta(days=1)

# Use the last available month's data for prediction (similar data structure as last_month_data)
last_month_data = X.tail(1)  # Using the most recent data as input for prediction
next_month_prediction = model.predict(last_month_data)
print("Predicted waste for next month (biodegradable, residual):", next_month_prediction)

# Display the date range for the next month
print(f"Next month prediction range: {next_month_start.date()} to {next_month_end.date()}")
