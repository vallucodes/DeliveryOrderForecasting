## 🎯 Project Overview

### Problem Statement

**Objective**: Estimate baseline hourly order volumes to optimize courier supply and maintain consistent delivery times across the Helsinki metropolitan area.

### Business Impact

This forecasting system supports critical operational decisions:

- **Dynamic Courier Incentives**: Automatically adjust pay and delivery frequency during peak demand
- **Anomaly Detection**: Rapidly identify unusual order patterns (±50% from baseline) to detect app issues or unexpected demand spikes
- **Customer Satisfaction**: Maintain consistent delivery times even during peak hours
- **Courier Retention**: Improve courier satisfaction through reliable, data-driven incentive systems

### Stakeholders

- **Operations Managers**: Use forecasts as baseline for workforce planning
- **Live Operations Team**: Monitor real-time order volumes against predictions
- **Uptime Monitoring**: Detect service disruptions through order volume anomalies

---

## 📁 Data

**Source**: `orders_spring_2022.csv`

**Key Findings from Exploration**:
- Clear cyclical patterns with weekday peaks at 8-9 AM and 3-4 PM
- Weekend peaks concentrated around 3-4 PM
- Progressive order growth Monday → Friday
- Rain increases order volume on weekdays
- 5 missing hourly slots (0 orders) between hours 7 and 20
- Some datapoints missing precipitation values

---

### Why GAM with Interaction was Selected

**Performance**:
- Competitive MAE across both weekday and weekend models
- Strong generalization (tight confidence intervals)
- Consistent performance between CV and test sets

**Interpretability**:
- Visual inspection of smooth functions for each feature
- Clear statistical outputs with confidence intervals
- Stakeholder-friendly explanations of predictions

**Operational Fit**:
- Captures non-linear temporal patterns critical for forecasting
- Interaction term models complex hour × day-of-week dynamics
- Stable predictions build trust with operations teams

**Trade-offs Considered**:
- XGBoost showed slightly lower CV error on weekdays but sacrifices interpretability
- Linear models underfit the non-linear patterns

---

## 🔧 Modeling Approach

### Architecture

**Separate Models**: Weekday and weekend models to capture distinct behavioral patterns

**Key Components**:
- **Non-linear splines**: Capture smooth, continuous patterns throughout the day
- **Tensor interaction**: `Hour` × `Day of week` interaction term captures complex temporal dependencies
- **Critical hours focus**: Models trained on 7:00-20:00 window for maximum operational relevance

### Features

**Selected Features**:
- `Hour of day` (1-24): Captures intraday cyclical patterns
- `Day of week` (Mon-Sun): Accounts for weekly variation
- `Precipitation` (mm): Weather-driven demand changes

**Excluded Features** (with rationale):
- *Order category & item count*: Order-level details don't predict volume
- *Delivery time estimates*: Reflect distance, not demand
- *Venue/customer location*: Insufficient data for location-specific models
- *Courier supply index*: Risk of data leakage (response variable, not predictor)

### Target Variable

`Order count per hour`: Primary metric for baseline forecasting

---

## 📈 Model Performance

### Evaluation Metrics

| Metric | Weekdays | Weekends |
|--------|----------|----------|
| **MAE** | 4.67 orders/hour | 4.70 orders/hour |
| **RMSE** | 6.05 orders/hour | 5.70 orders/hour |
| **MAPE** | 35.95% | 30.20% |
| **Pseudo R²** | 0.68 | 0.76 |

**Why MAE is the primary metric**:
- Prioritized over RMSE to focus on median performance
- Targets common scenarios rather than outliers
- Provides reliable baseline for operational planning
- ~4.7 orders/hour average error across both models

### Model Statistics

#### Weekday Model Feature Contributions

| Feature | EDoF | P-value |
|---------|------|---------|
| s(hour) | 7.3 | < 0.001 |
| s(day_of_week) | 3.7 | < 0.001 |
| s(precipitation) | 1.7 | < 0.001 |
| **te(hour, day_of_week)** | **14.2** | **< 0.001** |

The high interaction term EDoF = 14.2 confirms non-linear relationship between hour and day-of-week, justifying GAM over linear models.

#### Weekend Model Feature Contributions

| Feature | EDoF | P-value |
|---------|------|---------|
| s(hour) | 6.5 | < 0.001 |
| s(day_of_week) | 1.1 | 0.844 |
| s(precipitation) | 1.6 | 0.122 |
| **te(hour, day_of_week)** | **3.8** | **< 0.001** |

Despite individual features appearing non-significant, the strong interaction term (EDoF = 3.8, p < 0.001) captures non-linear relationship between hour and day-of-week.

### Assumptions

1. **Additive structure**: Order volume = sum of feature splines + interaction term
2. **Smooth effects**: Each feature contributes continuously (no sudden jumps)
3. **Independence**: No significant long-term seasonality (validated in EDA)
4. **Fixed rain effect**: Precipitation impact assumed constant across all hours

---

## ⚠️ Critical Considerations

### Error Types

**Over-prediction**:
- Leads to incentive fatigue among couriers
- Reduces effectiveness when incentives are genuinely needed
- Impacts customer experience during actual high-demand periods

**Under-prediction**:
- Insufficient courier supply
- Longer delivery times
- Degraded customer satisfaction

### Production Risks

**External Disruptions**:
- Holidays and special events
- Major shifts in retail vs. food delivery mix
- Competitor market share changes

**Seasonal Transitions**:
- Rapid seasonal changes may lack training data
- Slow transitions cause old patterns to influence new predictions

---

## 🚧 Known Limitations

1. **Baseline nature**: Cannot predict special events or sudden changes
2. **Average conditions**: Doesn't account for holidays or seasonal variations
3. **Delivery mode**: No distinction between bike, car, or other delivery types
4. **Granularity**: Area-level only (whole Helsinki), not location-specific
5. **Weekend weather**: Insufficient data for reliable rain effect on weekends
6. **Item details**: Disregards order composition and item counts

---

## 🔮 Future Improvements

### Priority Enhancements

1. **Location-Based Models**
   - Split forecasts by major hubs: Pasila, Itäkeskus, Kalasatama, Kamppi
   - Enable targeted courier attraction to specific high-demand zones

2. **Seasonal Adaptation**
   - Automatic retraining for seasonal models (Spring, Summer, Fall, Winter)
   - Rolling 3-4 month data window for continuous adaptation
   - Special holiday models for July and December

3. **Weekend Model Refinement**
   - Collect additional weekend data for precipitation analysis
   - Improve confidence intervals for weather effects
   - Explore transfer of weekday precipitation patterns to weekend model

4. **Hyperparameter Optimization**
   - Separate optimization for weekday vs. weekend models
   - Address high p-values in weekend feature terms

---

## 💼 Business Value

### Operational Benefits

- **Proactive Planning**: Baseline enables forward-looking courier management
- **Fast Response**: Rapid anomaly detection for system issues or demand spikes
- **Cost Efficiency**: Optimized incentive spending through accurate forecasting
- **Service Quality**: Consistent delivery times build brand reputation

### Trust & Adoption

Model usefulness depends on stakeholder trust:
- Operations managers rely on stable, interpretable predictions
- Couriers respond to reliable incentive systems
- Accurate baselines enable confident decision-making

---

## 📝 Model Card Summary

| Attribute | Value |
|-----------|-------|
| Model Type | Generalized Additive Model (GAM) |
| Prediction Target | Hourly order count |
| Geographic Scope | Helsinki metropolitan area |
| Temporal Resolution | 1 hour |
| Training Data | Spring 2022 orders (947 samples) |
| Training Split | Weekdays: 701 samples, Weekends: 246 samples |
| Primary Metric | MAE ≈ 4.7 orders/hour |
| Update Frequency | Manual (requires retraining) |
| Operational Hours | 7:00-20:00 |

---
