import pandas as pd
from itertools import product
from pygam import LinearGAM, s, te
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

HOURS_START = 7
HOURS_END = 20

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
hours_range = range(HOURS_START, HOURS_END + 1)

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
        s(0, n_splines=10) +      # Smooth function for hour
        s(1, n_splines=5) +       # Smooth function for day_of_week
        s(2, n_splines=5) +       # Smooth function for precipitation
        te(0, 1)                  # Interaction between hour and day_of_week
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


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FixedLocator, NullLocator

def plot_gam_terms(model, feature_names, is_weekend=False, label_size=19, tick_size=15):
    fig, axs = plt.subplots(1, len(feature_names), figsize=(15, 5))
    if len(feature_names) == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        if feature_names[i] == 'day of week':
            if is_weekend:
                unique_days = [5, 6]
            else:
                unique_days = [0, 1, 2, 3, 4]
            n_points = len(unique_days)
            XX = model.generate_X_grid(term=i, n=n_points)
            XX[:, i] = unique_days
            ax.set_xticks(unique_days)
            ax.set_xlim(min(unique_days) - 0.25, max(unique_days) + 0.25)
            ax.xaxis.set_major_locator(FixedLocator(unique_days))
            ax.xaxis.set_minor_locator(NullLocator())
            pdep, confi = model.partial_dependence(term=i, X=XX, width=0.95)
            for j, day in enumerate(unique_days):
                ax.hlines(pdep[j], day - 0.2, day + 0.2, colors='C0')
                ax.hlines(confi[j, 0], day - 0.2, day + 0.2, colors='r', linestyles='--')
                ax.hlines(confi[j, 1], day - 0.2, day + 0.2, colors='r', linestyles='--')
        else:
            XX = model.generate_X_grid(term=i)
            if feature_names[i] == 'hour':
                ax.xaxis.set_major_locator(MultipleLocator(2))
                ax.set_xlim(6, 21)
            pdep, confi = model.partial_dependence(term=i, X=XX, width=0.95)
            ax.plot(XX[:, i], pdep)
            ax.plot(XX[:, i], confi[:, 0], c='r', ls='--')
            ax.plot(XX[:, i], confi[:, 1], c='r', ls='--')
        p_val = model.statistics_['p_values'][i]
        if p_val < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p_val:.3f}'
        ax.text(0.05, 0.95, p_text, transform=ax.transAxes, va='top', fontsize=15)
        ax.set_xlabel(feature_names[i], fontsize=label_size)
        ax.set_ylabel("Effect on Order Count", fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
    plt.tight_layout()
    plt.show()

# Plot the features
plot_gam_terms(weekday_model, ['hour', 'day of week', 'precipitation (mm)'])
plot_gam_terms(weekend_model, ['hour', 'day of week', 'precipitation (mm)'], is_weekend=True)

print(weekday_model.summary())
print(weekend_model.summary())
