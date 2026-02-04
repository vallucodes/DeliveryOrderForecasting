from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from pygam import LinearGAM, s, te, f
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy import stats

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

def get_models(xgb_params):
    """Define models with their fit/predict interfaces."""
    return {
        "Dummy (Mean)": DummyRegressor(strategy='mean'),
        "Linear Regression": LinearRegression(),
        "GAM (additive)": LinearGAM(s(0, n_splines=10) + s(1, n_splines=5) + s(2, n_splines=5)),
        "GAM (with interaction)": LinearGAM(s(0, n_splines=10) + s(1, n_splines=5) + s(2, n_splines=5) + te(0, 1)),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=4),
        "XGBoost": XGBRegressor(**xgb_params),
    }

def prepare_features(X, for_linear=False, reference_columns=None):
    """Prepare features: tree/GAM use raw, linear uses one-hot. Optionally reindex to match reference."""
    if for_linear:
        X_prep = pd.get_dummies(X, columns=['hour', 'day_of_week'], drop_first=True, dtype=int)
        if reference_columns is not None:
            X_prep = X_prep.reindex(columns=reference_columns, fill_value=0)
        return X_prep
    return X[['hour', 'day_of_week', 'precipitation']].values

def train_eval_single_split(models, X_train_tree, X_val_tree, X_train_lin, X_val_lin, y_train, y_val):
    """Train and eval all models on one split."""
    results = {}
    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(X_train_lin, y_train)
            preds = model.predict(X_val_lin)
        else:
            model.fit(X_train_tree, y_train)
            preds = model.predict(X_val_tree)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        results[name] = {'rmse': rmse, 'mae': mae}
    return results

def summarize_cv(results):
    """Compute mean/std/CI from CV results."""
    summary = {}
    for name in results:
        rmses = [r['rmse'] for r in results[name]]
        summary[name] = {
            'mean_rmse': np.mean(rmses),
            'std_rmse': np.std(rmses),
            'ci_rmse': [np.percentile(rmses, 2.5), np.percentile(rmses, 97.5)]
        }
    return summary

def final_eval(models, X_dev_tree, X_final_tree, X_dev_lin, X_final_lin, y_dev, y_final):
    """Train on full dev, eval on final test."""
    final_results = {}
    for name, model in models.items():
        if name == "Linear Regression":
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

def train_evaluate_all_models(X, y, group_name, n_seeds=30):
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

    # Step 1: Hold out final test set
    X_dev, X_final, y_dev, y_final = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=999)
    print(f"\n{'='*80}\n{group_name} - Data Split\n{'='*80}")
    print(f"Development set: {len(X_dev)} samples\nFinal test set: {len(X_final)} samples")

    # Pre-encode full dev for efficiency
    X_dev_tree = prepare_features(X_dev)
    X_dev_lin = prepare_features(X_dev, for_linear=True)
    X_final_tree = prepare_features(X_final)
    X_final_lin = prepare_features(X_final, for_linear=True, reference_columns=X_dev_lin.columns)  # Added reference_columns to reindex

    # Step 2: MCCV on dev set
    print(f"\n{'='*80}\n{group_name} - Cross-validation with {n_seeds} splits\n{'='*80}")
    cv_results = {name: [] for name in models}
    cv_rmses = {name: [] for name in models}  # For stats tests
    for seed in range(n_seeds):
        # Split dev into train/val
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_dev, y_dev, test_size=0.2, random_state=seed)
        # Prep split features
        X_tr_tree = prepare_features(X_tr)
        X_vl_tree = prepare_features(X_vl)
        X_tr_lin = prepare_features(X_tr, for_linear=True)
        X_vl_lin = prepare_features(X_vl, for_linear=True, reference_columns=X_tr_lin.columns)  # Use reference for val
        # Note: Removed extra reindex line, now handled in prepare_features

        split_results = train_eval_single_split(models, X_tr_tree, X_vl_tree, X_tr_lin, X_vl_lin, y_tr, y_vl)
        for name, metrics in split_results.items():
            cv_results[name].append(metrics)
            cv_rmses[name].append(metrics['rmse'])

    # Step 3: Report CV summary and stats
    summary = summarize_cv(cv_results)
    print(f"\n--- {group_name} Cross-Validation Results ---")
    print(f"{'Model':<25} {'Mean RMSE':<12} {'Std RMSE':<12} {'95% CI RMSE':<25}")
    print("-" * 85)
    for name, stats in summary.items():
        print(f"{name:<25} {stats['mean_rmse']:<12.2f} {stats['std_rmse']:<12.2f} [{stats['ci_rmse'][0]:.2f}, {stats['ci_rmse'][1]:.2f}]")

    # Step 4: Final eval on held-out
    print(f"\n{'='*80}\n{group_name} - FINAL TEST on Held-Out Data\n{'='*80}")
    final_results = final_eval(models, X_dev_tree, X_final_tree, X_dev_lin, X_final_lin, y_dev, y_final)
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
    print("-" * 60)
    for name, metrics in final_results.items():
        print(f"{name:<25} {metrics['rmse']:<10.2f} {metrics['mae']:<10.2f} {metrics['mape']:<10.2f}")

    return cv_results

# Run analysis
X_weekdays = weekdays[['hour', 'day_of_week', 'precipitation']]
y_weekdays = weekdays['order_count']
weekday_results = train_evaluate_all_models(X_weekdays, y_weekdays, "Weekdays")
X_weekends = weekends[['hour', 'day_of_week', 'precipitation']]
y_weekends = weekends['order_count']
weekend_results = train_evaluate_all_models(X_weekends, y_weekends, "Weekends")
