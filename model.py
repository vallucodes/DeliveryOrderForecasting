from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv('orders_spring_2022.csv')

# Convert the timestamp string to datetime
df['order_placed_at_utc'] = pd.to_datetime(df['order_placed_at_utc'])

# Extract time and precipitation features
df['date'] = df['order_placed_at_utc'].dt.date
df['hour'] = df['order_placed_at_utc'].dt.hour
df['day_of_week'] = df['order_placed_at_utc'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Saturday, Sunday

# Group data together for each hour
hourly = df.groupby(['date', 'hour']).agg({
    'order_placed_at_utc': 'count',  # This counts orders per hour
    'precipitation': 'first',  # Assuming same precipitation for the whole hour
    'day_of_week': 'first',  # Keep day of week
    'is_weekend': 'first',  # Weekend flag
}).reset_index()

# Rename the count column
hourly.rename(columns={'order_placed_at_utc': 'order_count'}, inplace=True)

# Sort by date and hour
hourly_sorted = hourly.sort_values(['date', 'hour']).reset_index(drop=True)

# Remove NaNs
hourly_sorted = hourly_sorted.dropna(subset=['hour', 'day_of_week', 'is_weekend', 'precipitation', 'order_count']).reset_index(drop=True)

# Split into weekdays and weekends
weekdays = hourly_sorted[hourly_sorted['is_weekend'] == 0]
weekends = hourly_sorted[hourly_sorted['is_weekend'] == 1]

# Features (dropping 'is_weekend' since constant in each subset; using continuous precipitation)
features = ['hour', 'day_of_week', 'precipitation']

# Function to train and evaluate models (updated to include test eval)
def train_evaluate(X, y, group_name):
    if len(X) == 0:
        print(f"No data for {group_name}")
        return

    split_1 = int(len(X) * 0.7)  # 70% train
    split_2 = int(len(X) * 0.85)  # 85% train+validation

    X_train, y_train = X.iloc[:split_1], y.iloc[:split_1]
    X_val, y_val = X.iloc[split_1:split_2], y.iloc[split_1:split_2]
    X_test, y_test = X.iloc[split_2:], y.iloc[split_2:]

    print(f"{group_name} - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # Train models
    models = {
        "Poisson": PoissonRegressor(max_iter=200),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    # Helper to get metrics
    def get_metrics(preds, y_true):
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        neg = (preds < 0).sum()
        return rmse, mae, neg

    # Validation results (for model comparison)
    print(f"\n--- {group_name} Validation Results ---")
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Negative Preds'}")
    print("-" * 55)
    val_results = {}
    for name, model in models.items():
        preds = model.predict(X_val)
        rmse, mae, neg = get_metrics(preds, y_val)
        mape = np.mean(np.abs((y_val - preds) / (y_val + 1e-10))) * 100
        val_results[name] = rmse
        print(f"{name:<25} {rmse:<10.2f} {mae:<10.2f} {mape:<10.2f} {neg}")

    # Select best model based on val RMSE (for demo; you could choose manually)
    best_model_name = min(val_results, key=val_results.get)
    print(f"\nBest model on validation for {group_name}: {best_model_name}")

    # Test results (final unbiased evaluation for all models)
    print(f"\n--- {group_name} Test Results ---")
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Negative Preds'}")
    print("-" * 55)
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse, mae, neg = get_metrics(preds, y_test)
        mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-10))) * 100
        print(f"{name:<25} {rmse:<10.2f} {mae:<10.2f} {mape:<10.2f} {neg}")

# For weekdays
X_weekdays = weekdays[features]
y_weekdays = weekdays['order_count']
train_evaluate(X_weekdays, y_weekdays, "Weekdays")

# For weekends
X_weekends = weekends[features]
y_weekends = weekends['order_count']
train_evaluate(X_weekends, y_weekends, "Weekends")
