import pandas as pd
import numpy as np
from itertools import product
from sklearn.linear_model import LinearRegression
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

# Convert date to datetime for complete range creation
hourly['date'] = pd.to_datetime(hourly['date'])

# Create complete date-hour range
min_date = hourly['date'].min()
max_date = hourly['date'].max()
all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
hours_7_20 = range(7, 21)

# Create all combinations for dates and selected hours
all_combinations = pd.DataFrame(list(product(all_dates, hours_7_20)),
                                columns=['date', 'hour'])

# Merge to fill missing hours
hourly_complete = all_combinations.merge(hourly, on=['date', 'hour'], how='left')

# Fill ONLY the missing order_count with 0 (the 5 missing hours)
hourly_complete['order_count'] = hourly_complete['order_count'].fillna(0).astype(int)

# Derive day_of_week and is_weekend from date for any missing rows
hourly_complete['day_of_week'] = hourly_complete['date'].dt.dayofweek
hourly_complete['is_weekend'] = hourly_complete['day_of_week'].isin([5, 6]).astype(int)

hourly_sorted = hourly_complete.sort_values(['date', 'hour']).reset_index(drop=True)

# Drop rows where precipitation is NaN
hourly_filtered = hourly_sorted.dropna(subset=['precipitation']).reset_index(drop=True)

# Split into weekdays and weekends BEFORE one-hot encoding
weekdays = hourly_filtered[hourly_filtered['is_weekend'] == 0].copy()
weekends = hourly_filtered[hourly_filtered['is_weekend'] == 1].copy()

# One-hot encode separately
weekdays = pd.get_dummies(weekdays, columns=['hour', 'day_of_week'], drop_first=True, dtype=int)
weekends = pd.get_dummies(weekends, columns=['hour', 'day_of_week'], drop_first=True, dtype=int)

# Get feature lists
weekday_features = [col for col in weekdays.columns if col.startswith(('hour_', 'day_of_week_')) or col == 'precipitation']
weekend_features = [col for col in weekends.columns if col.startswith(('hour_', 'day_of_week_')) or col == 'precipitation']

def train_model(X, y):
    model = LinearRegression()
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

def run_pipeline(data, features, group_label):
    if len(data) == 0:
        return None

    # 1. Split Data into train and test
    X = data[features]
    y = data['order_count']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=999
    )

    # 2. Train Linear Regression
    model = train_model(X_train, y_train)

    # 3. Evaluate
    evaluate_model(model, X_test, y_test, group_label)

    # 4. Print feature importances (coefficients)
    print(f"\nTop 10 Most Important Features (by absolute coefficient):")
    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_
    })
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    print(coef_df[['feature', 'coefficient']].head(10))

    return model

# Execute
print("\n" + "="*60)
print("WEEKDAYS MODEL")
print("="*60)
weekday_model = run_pipeline(weekdays, weekday_features, "Weekdays")

print("\n" + "="*60)
print("WEEKENDS MODEL")
print("="*60)
weekend_model = run_pipeline(weekends, weekend_features, "Weekends")


# Optional: Plot feature importances
import matplotlib.pyplot as plt

def plot_top_coefficients(model, features, title, top_n=15):
    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_
    })
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False).head(top_n)
    coef_df = coef_df.sort_values('coefficient', ascending=True)

    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in coef_df['coefficient']]
    plt.barh(range(len(coef_df)), coef_df['coefficient'], color=colors)
    plt.yticks(range(len(coef_df)), coef_df['feature'])
    plt.xlabel('Coefficient Value')
    plt.title(title)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()

plot_top_coefficients(weekday_model, weekday_features, "Weekday Model - Top 15 Features by Importance")
plot_top_coefficients(weekend_model, weekend_features, "Weekend Model - Top 15 Features by Importance")
