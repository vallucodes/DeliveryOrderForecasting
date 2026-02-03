from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
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

# Filter to only hours from 7 to 20
hourly_filtered = hourly_sorted[(hourly_sorted['hour'] >= 7) & (hourly_sorted['hour'] <= 20)].reset_index(drop=True)

# Remove NaNs
hourly_filtered = hourly_filtered.dropna(subset=['hour', 'day_of_week', 'is_weekend', 'precipitation', 'order_count']).reset_index(drop=True)

# One-hot encode categorical features: hour and day_of_week
hourly_filtered = pd.get_dummies(hourly_filtered, columns=['hour', 'day_of_week'], drop_first=True, dtype=int)

# Split into weekdays and weekends (is_weekend is still available)
weekdays = hourly_filtered[hourly_filtered['is_weekend'] == 0]
weekends = hourly_filtered[hourly_filtered['is_weekend'] == 1]

# Updated features: now includes one-hot columns, drops originals and is_weekend
features = [col for col in hourly_filtered.columns if col.startswith(('hour_', 'day_of_week_')) or col == 'precipitation']

# Function to train and evaluate models
def train_evaluate(X, y, group_name):
    if len(X) == 0:
        print(f"No data for {group_name}")
        return

    # Random split: 70% train, then split remaining 30% into 15% val + 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=4, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=4, shuffle=True)

    # print(f"{group_name} - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    # print(f"Features used ({len(features)} total): {features}")

    # Print stats to verify randomness/balance
    print(f"\n{group_name} Split Stats:")
    for split_name, split_X, split_y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        mean_precip = split_X['precipitation'].mean()
        std_precip = split_X['precipitation'].std()
        mean_orders = split_y.mean()
        std_orders = split_y.std()
        print(f"{split_name}: Mean Precip {mean_precip:.2f} (std {std_precip:.2f}), Mean Orders {mean_orders:.2f} (std {std_orders:.2f})")

    # Train models (added Dummy as baseline)
    models = {
        "Dummy (Mean)": DummyRegressor(strategy='mean'),  # Baseline model
        "Poisson": PoissonRegressor(max_iter=200),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=4),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=4)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    # Helper to get metrics
    def get_metrics(preds, y_true):
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        neg = (preds < 0).sum()
        mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-10))) * 100
        return rmse, mae, neg, mape

    # Validation results (for model comparison)
    print(f"\n--- {group_name} Validation Results ---")
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Negative Preds'}")
    print("-" * 75)
    val_results = {}
    for name, model in models.items():
        preds = model.predict(X_val)
        rmse, mae, neg, mape = get_metrics(preds, y_val)
        val_results[name] = rmse
        print(f"{name:<25} {rmse:<10.2f} {mae:<10.2f} {mape:<10.2f} {neg}")

    # Select best model based on val RMSE
    best_model_name = min(val_results, key=val_results.get)
    print(f"\nBest model on validation for {group_name}: {best_model_name}")

    # Test results (final unbiased evaluation for all models)
    print(f"\n--- {group_name} Test Results ---")
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Negative Preds'}")
    print("-" * 75)
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse, mae, neg, mape = get_metrics(preds, y_test)
        print(f"{name:<25} {rmse:<10.2f} {mae:<10.2f} {mape:<10.2f} {neg}")

# For weekdays
X_weekdays = weekdays[features]
y_weekdays = weekdays['order_count']
train_evaluate(X_weekdays, y_weekdays, "Weekdays")

# For weekends
X_weekends = weekends[features]
y_weekends = weekends['order_count']
train_evaluate(X_weekends, y_weekends, "Weekends")
