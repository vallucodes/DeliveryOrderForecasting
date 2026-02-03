from sklearn.linear_model import PoissonRegressor, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
hourly_filtered = hourly_sorted[(hourly_sorted['hour'] >= 7) & (hourly_sorted['hour'] <= 20)].reset_index(drop=True)

# Remove NaNs
hourly_filtered = hourly_filtered.dropna(subset=['hour', 'day_of_week', 'precipitation', 'order_count']).reset_index(drop=True)

# One-hot encode categorical features: hour and day_of_week
hourly_filtered = pd.get_dummies(hourly_filtered, columns=['hour', 'day_of_week'], drop_first=True, dtype=int)

# Split into weekdays and weekends (is_weekend is still available)
weekdays = hourly_filtered[hourly_filtered['is_weekend'] == 0].copy()
weekends = hourly_filtered[hourly_filtered['is_weekend'] == 1].copy()

# Updated features: now includes one-hot columns, drops originals and is_weekend
features = [col for col in hourly_filtered.columns if col.startswith(('hour_', 'day_of_week_')) or col == 'precipitation']

# Function to train and evaluate models
def train_evaluate(X, y, group_name):
    if len(X) == 0:
        print(f"No data for {group_name}")
        return

    # 1. SPLIT DATA FIRST (Before CV)
    # Random split: 70% train, then split remaining 30% into 15% val + 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

    print(f"\n{group_name} Split Stats:")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 5-fold cross validation
    kf = KFold(n_splits=30, shuffle=True, random_state=42)

    models = {
        "Dummy (Mean)": DummyRegressor(strategy='mean'),
        "Linear Regression": LinearRegression(),
        "Poisson": PoissonRegressor(max_iter=200),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=4),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=4)
    }

    print(f"\n--- {group_name} Cross-Validation (on Train set only) ---")
    print(f"{'Model':<25} {'Mean RMSE':<12} {'Std RMSE':<12}")
    print("-" * 50)

    # Dictionary to store fitted models so we don't have to refit them for Test
    fitted_models = {}

    for name, model in models.items():
        # 2. RUN CV ON TRAIN SET ONLY
        # This prevents "Data Leakage" (peeking at test data)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
        rmse_scores = -scores
        print(f"{name:<25} {rmse_scores.mean():<12.2f} {rmse_scores.std():<12.2f}")

        # 3. FIX: EXPLICITLY FIT THE MODEL
        # cross_val_score didn't fit the model instance, so we must do it here
        model.fit(X_train, y_train)
        fitted_models[name] = model

    # Helper to get metrics
    def get_metrics(preds, y_true):
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        neg = (preds < 0).sum()
        # Avoid division by zero
        mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-10))) * 100
        return rmse, mae, neg, mape

    # 4. VALIDATION RESULTS
    print(f"\n--- {group_name} Validation Results ---")
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'NegPreds'}")
    print("-" * 75)

    val_results = {}
    for name, model in fitted_models.items():
        preds = model.predict(X_val)
        rmse, mae, neg, mape = get_metrics(preds, y_val)
        val_results[name] = rmse
        print(f"{name:<25} {rmse:<10.2f} {mae:<10.2f} {mape:<10.2f} {neg}")

    # Select best model based on val RMSE
    best_model_name = min(val_results, key=val_results.get)
    print(f"\nBest model on validation for {group_name}: {best_model_name}")

    # 5. TEST RESULTS
    print(f"\n--- {group_name} Test Results (Final Check) ---")
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'NegPreds'}")
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
