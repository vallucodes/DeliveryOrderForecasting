import pandas as pd
from itertools import product
from pygam import LinearGAM, s, te
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

HOUR_START = 7
HOUR_END = 20

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

# Convert date to datetime
hourly['date'] = pd.to_datetime(hourly['date'])

# Create full date and hour range
min_date = hourly['date'].min()
max_date = hourly['date'].max()
all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
hours_range = range(HOUR_START, HOUR_END + 1)

# Create all combinations date-hours
all_combinations = pd.DataFrame(list(product(all_dates, hours_range)),
                                columns=['date', 'hour'])

# Merge to fill missing hours
hourly_complete = all_combinations.merge(hourly, on=['date', 'hour'], how='left')

# Fill missing order_count with 0
hourly_complete['order_count'] = hourly_complete['order_count'].fillna(0).astype(int)

# Derive day_of_week and is_weekend from date for any missing rows
hourly_complete['day_of_week'] = hourly_complete['date'].dt.dayofweek
hourly_complete['is_weekend'] = hourly_complete['day_of_week'].isin([5, 6]).astype(int)

hourly_sorted = hourly_complete.sort_values(['date', 'hour']).reset_index(drop=True)

# Drop rows where precipitation is NaN
hourly_filtered = hourly_sorted.dropna(subset=['precipitation']).reset_index(drop=True)

# Split into weekdays and weekends
weekdays = hourly_filtered[hourly_filtered['is_weekend'] == 0].copy()
weekends = hourly_filtered[hourly_filtered['is_weekend'] == 1].copy()

def train_model(X, y):
    model = LinearGAM(
        s(0, n_splines=10) +
        s(1, n_splines=5) +
        s(2, n_splines=5) +
        te(0, 1)
    )
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test, group_name):
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    mape = (abs((y_test - preds) / (y_test + 1e-10))).mean() * 100

    print(f"\n{'='*30}\n{group_name} - Evaluation\n{'='*30}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {"rmse": rmse, "mae": mae, "mape": mape}

def run_pipeline(data, group_label):
    if len(data) == 0:
        return None

    X = data[['hour', 'day_of_week', 'precipitation']]
    y = data['order_count']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=999
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test, group_label)

    return model

# Execute
weekday_model = run_pipeline(weekdays, "Weekdays")
weekend_model = run_pipeline(weekends, "Weekends")
