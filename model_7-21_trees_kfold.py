from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
hourly_filtered = hourly_sorted[(hourly_sorted['hour'] >= 7) & (hourly_sorted['hour'] <= 21)].reset_index(drop=True)

# Remove NaNs
hourly_filtered = hourly_filtered.dropna(subset=['hour', 'day_of_week', 'precipitation', 'order_count']).reset_index(drop=True)

# No one-hot encoding - Keep categorical features as integers for tree models

# Split into weekdays and weekends
weekdays = hourly_filtered[hourly_filtered['is_weekend'] == 0].copy()
weekends = hourly_filtered[hourly_filtered['is_weekend'] == 1].copy()

# Features: keep hour, day_of_week, and precipitation as-is (no encoding)
features = ['hour', 'day_of_week', 'precipitation']

# Function to train and evaluate models
def train_evaluate(X, y, group_name):
    if len(X) == 0:
        print(f"No data for {group_name}")
        return

    # Random split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Print stats to verify randomness/balance
    print(f"\n{group_name} Split Stats:")
    for split_name, split_X, split_y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        mean_precip = split_X['precipitation'].mean()
        std_precip = split_X['precipitation'].std()
        mean_orders = split_y.mean()
        std_orders = split_y.std()
        print(f"{split_name}: Mean Precip {mean_precip:.2f} (std {std_precip:.2f}), Mean Orders {mean_orders:.2f} (std {std_orders:.2f})")

    print(f"\n{group_name} Split Stats:")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # KFold cross-validation setup
    kf = KFold(n_splits=15, shuffle=True, random_state=42)

    # Train tree-based models only
    models = {
        "Dummy (Mean)": DummyRegressor(strategy='mean'),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=4),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=4),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=4, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=4, verbosity=-1)
    }

    # Cross-validation on training set
    print(f"\n--- {group_name} Cross-Validation ---")
    print(f"{'Model':<25} {'Mean RMSE':<12} {'Std RMSE':<12}")
    print("-" * 50)

    fitted_models = {}
    for name, model in models.items():
        # Perform cross-validation on training set only
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
        rmse_scores = -scores  # Convert negative RMSE to positive
        print(f"{name:<25} {rmse_scores.mean():<12.2f} {rmse_scores.std():<12.2f}")

        # Fit model on full training set for later evaluation
        model.fit(X_train, y_train)
        fitted_models[name] = model

    # Helper to get metrics
    def get_metrics(preds, y_true):
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        neg = (preds < 0).sum()
        mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-10))) * 100
        return rmse, mae, neg, mape

    # Test results (final unbiased evaluation for all models)
    print(f"\n--- {group_name} Test Results on Unseen Values ---")
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Negative Preds'}")
    print("-" * 75)
    for name, model in fitted_models.items():
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
