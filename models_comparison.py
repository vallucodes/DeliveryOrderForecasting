from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from pygam import LinearGAM, s, te
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from itertools import product

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

# Create complete date-hour range
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

def get_models(xgb_params):
    return {
        "Dummy (Mean)": DummyRegressor(strategy='mean'),
        "Linear Regression": LinearRegression(),
        "Poisson": PoissonRegressor(max_iter=200),
        "GAM (additive)": LinearGAM(s(0, n_splines=10) + s(1, n_splines=5) + s(2, n_splines=5)),
        "GAM (with interaction)": LinearGAM(s(0, n_splines=10) + s(1, n_splines=5) + s(2, n_splines=5) + te(0, 1)),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=4),
        "XGBoost": XGBRegressor(**xgb_params),
    }

def prepare_features(X, for_linear=False, reference_columns=None):
    if for_linear:
        X_prep = pd.get_dummies(X, columns=['hour', 'day_of_week'], drop_first=True, dtype=int)
        if reference_columns is not None:
            X_prep = X_prep.reindex(columns=reference_columns, fill_value=0)
        return X_prep
    return X[['hour', 'day_of_week', 'precipitation']]

def train_eval_single_split(models, X_train_tree, X_test_tree, X_train_lin, X_test_lin, y_train, y_test):
    results = {}
    for name, model in models.items():
        if name in ["Linear Regression", "Poisson"]:
            model.fit(X_train_lin, y_train)
            preds = model.predict(X_test_lin)
        else:
            model.fit(X_train_tree, y_train)
            preds = model.predict(X_test_tree)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        results[name] = {'rmse': rmse, 'mae': mae}
    return results

def summarize_cv(results):
    summary = {}
    for name in results:
        rmses = [r['rmse'] for r in results[name]]
        maes = [r['mae'] for r in results[name]]
        summary[name] = {
            'mean_rmse': np.mean(rmses),
            'std_rmse': np.std(rmses),
            'ci_rmse': [np.percentile(rmses, 2.5), np.percentile(rmses, 97.5)],
            'mean_mae': np.mean(maes),
            'std_mae': np.std(maes),
            'ci_mae': [np.percentile(maes, 2.5), np.percentile(maes, 97.5)]
        }
    return summary

def final_eval(models, X_dev_tree, X_final_tree, X_dev_lin, X_final_lin, y_dev, y_final):
    final_results = {}
    for name, model in models.items():
        if name in ["Linear Regression", "Poisson"]:
            model.fit(X_dev_lin, y_dev)
            preds = model.predict(X_final_lin)
        else:
            model.fit(X_dev_tree, y_dev)
            preds = model.predict(X_final_tree)
        rmse = np.sqrt(mean_squared_error(y_final, preds))
        mae = mean_absolute_error(y_final, preds)
        mape = np.mean(np.abs((y_final - preds) / (y_final + 1e-10))) * 100
        final_results[name] = {'rmse': rmse, 'mae': mae, 'mape': mape}
    return final_results

def run_pipeline(X, y, group_name, n_seeds=30):
    if len(X) == 0:
        print(f"No data for {group_name}")
        return
    xgb_params = {
        'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': 4, 'verbosity': 0
    }
    models = get_models(xgb_params)

    # Split data into development and final evaluation sets
    X_dev, X_test_final, y_dev, y_test_final = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=999)
    print(f"\n{'='*80}\n{group_name} - Data Split\n{'='*80}")
    print(f"Development set: {len(X_dev)} samples\nFinal test set: {len(X_test_final)} samples")

    X_dev_tree = prepare_features(X_dev)
    X_dev_lin = prepare_features(X_dev, for_linear=True)
    X_final_tree = prepare_features(X_test_final)
    X_final_lin = prepare_features(X_test_final, for_linear=True, reference_columns=X_dev_lin.columns)

    # MCCV on dev set
    print(f"\n{'='*80}\n{group_name} - Cross-validation with {n_seeds} splits\n{'='*80}")
    cv_results = {name: [] for name in models}
    cv_rmses = {name: [] for name in models}
    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.2, random_state=seed)
        X_train_tree = prepare_features(X_train)
        X_test_tree = prepare_features(X_test)
        X_train_lin = prepare_features(X_train, for_linear=True)
        X_test_lin = prepare_features(X_test, for_linear=True, reference_columns=X_train_lin.columns)
        split_results = train_eval_single_split(models, X_train_tree, X_test_tree, X_train_lin, X_test_lin, y_train, y_test)
        for name, metrics in split_results.items():
            cv_results[name].append(metrics)
            cv_rmses[name].append(metrics['rmse'])

    # Report CV summary and stats
    summary = summarize_cv(cv_results)
    print(f"\n--- {group_name} Cross-Validation Results ---")
    print(f"{'Model':<25} {'Mean MAE':<12} {'Std MAE':<12} {'95% CI MAE':<20} {'Mean RMSE':<12} {'Std RMSE':<12} {'95% CI RMSE':<20}")
    print("-" * 115)
    for name, stats in summary.items():
        ci_mae_str = f"[{stats['ci_mae'][0]:.2f}, {stats['ci_mae'][1]:.2f}]"
        ci_rmse_str = f"[{stats['ci_rmse'][0]:.2f}, {stats['ci_rmse'][1]:.2f}]"
        print(f"{name:<25} {stats['mean_mae']:<12.2f} {stats['std_mae']:<12.2f} {ci_mae_str:<20} {stats['mean_rmse']:<12.2f} {stats['std_rmse']:<12.2f} {ci_rmse_str:<20}")

    # Final eval on test data
    print(f"\n{'='*80}\n{group_name} - Models run on test data\n{'='*80}")
    final_results = final_eval(models, X_dev_tree, X_final_tree, X_dev_lin, X_final_lin, y_dev, y_test_final)
    print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 60)
    for name, metrics in final_results.items():
        print(f"{name:<25} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f} {metrics['mape']:<10.2f}")

    return cv_results

# Run analysis
X_weekdays = weekdays[['hour', 'day_of_week', 'precipitation']]
y_weekdays = weekdays['order_count']
weekday_results = run_pipeline(X_weekdays, y_weekdays, "Weekdays")
X_weekends = weekends[['hour', 'day_of_week', 'precipitation']]
y_weekends = weekends['order_count']
weekend_results = run_pipeline(X_weekends, y_weekends, "Weekends")
