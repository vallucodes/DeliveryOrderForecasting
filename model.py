import pandas as pd
import numpy as np
from pygam import LinearGAM, s, te
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load and prepare data
df = pd.read_csv('orders_spring_2022.csv')
df['order_placed_at_utc'] = pd.to_datetime(df['order_placed_at_utc'])
df['date'] = df['order_placed_at_utc'].dt.date
df['hour'] = df['order_placed_at_utc'].dt.hour
df['day_of_week'] = df['order_placed_at_utc'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Group by hour
hourly = df.groupby(['date', 'hour']).agg({
    'order_placed_at_utc': 'count',
    'precipitation': 'first',
    'day_of_week': 'first',
    'is_weekend': 'first',
}).reset_index()
hourly.rename(columns={'order_placed_at_utc': 'order_count'}, inplace=True)
hourly_sorted = hourly.sort_values(['date', 'hour']).reset_index(drop=True)
hourly_filtered = hourly_sorted[(hourly_sorted['hour'] >= 7) & (hourly_sorted['hour'] <= 21)].reset_index(drop=True)
hourly_filtered = hourly_filtered.dropna(subset=['hour', 'day_of_week', 'precipitation', 'order_count']).reset_index(drop=True)

# Split into weekdays and weekends
weekdays = hourly_filtered[hourly_filtered['is_weekend'] == 0].copy()
weekends = hourly_filtered[hourly_filtered['is_weekend'] == 1].copy()

def prepare_features(df, features_list):
    X = df[features_list].values  # Convert to numpy array for GAM
    return X

def train_model(X, y):
    model = LinearGAM(
        s(0, n_splines=10) +      # Smooth function for hour
        s(1, n_splines=5) +       # Smooth function for day_of_week
        s(2, n_splines=5) +       # Smooth function for precipitation
        te(0, 1)                  # Interaction between hour and day_of_week
    )
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test, group_name):
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-10))) * 100

    print(f"\n{'='*30}\n{group_name} - Evaluation\n{'='*30}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {"rmse": rmse, "mae": mae, "mape": mape}

def run_pipeline(data, group_label):
    if len(data) == 0:
        return None

    # 1. Split Data
    X = data[['hour', 'day_of_week', 'precipitation']]
    y = data['order_count']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=999
    )

    # 2. Prepare Features (no encoding for GAM)
    X_train = prepare_features(X_train, ['hour', 'day_of_week', 'precipitation'])
    X_test = prepare_features(X_test, ['hour', 'day_of_week', 'precipitation'])

    # 3. Train GAM
    model = train_model(X_train, y_train)

    # 4. Evaluate
    evaluate_model(model, X_test, y_test, group_label)

    return model

# Execute
weekday_model = run_pipeline(weekdays, "Weekdays")
weekend_model = run_pipeline(weekends, "Weekends")
