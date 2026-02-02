import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('orders_spring_2022.csv')  # Replace with your filename

# Convert the timestamp string to datetime
df['order_placed_at_utc'] = pd.to_datetime(df['order_placed_at_utc'])

# Extract time features
df['date'] = df['order_placed_at_utc'].dt.date
df['hour'] = df['order_placed_at_utc'].dt.hour
df['day_of_week'] = df['order_placed_at_utc'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Saturday, Sunday
df['has_rain'] = (df['precipitation'] > 0).astype(int)

# Aggregate to hourly counts
hourly = df.groupby(['date', 'hour']).agg({
    'order_placed_at_utc': 'count',  # This counts orders per hour
    'precipitation': 'first',  # Assuming same precipitation for the whole hour
    'courier_supply_index': 'mean',  # Average courier supply in that hour
    'day_of_week': 'first',  # Keep day of week
    'is_weekend': 'first',  # Keep weekend flag
    'has_rain': 'first'
}).reset_index()

# Rename the count column
hourly.rename(columns={'order_placed_at_utc': 'order_count'}, inplace=True)

# Create 4 different groups
weekday_no_rain = hourly[(hourly['day_of_week'] < 5) & (hourly['has_rain'] == 0)].groupby('hour')['order_count'].mean().reset_index()
weekday_rain = hourly[(hourly['day_of_week'] < 5) & (hourly['has_rain'] == 1)].groupby('hour')['order_count'].mean().reset_index()
weekend_no_rain = hourly[(hourly['day_of_week'] >= 5) & (hourly['has_rain'] == 0)].groupby('hour')['order_count'].mean().reset_index()
weekend_rain = hourly[(hourly['day_of_week'] >= 5) & (hourly['has_rain'] == 1)].groupby('hour')['order_count'].mean().reset_index()

# # Rename columns
for dataset in [weekday_no_rain, weekday_rain, weekend_no_rain, weekend_rain]:
    dataset.columns = ['hour', 'avg_order_count']

# Create 2x2 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Weekdays: Rain vs No Rain
ax1.plot(weekday_no_rain['hour'], weekday_no_rain['avg_order_count'],
         marker='o', linewidth=2, label='No Rain', color='steelblue')
ax1.plot(weekday_rain['hour'], weekday_rain['avg_order_count'],
         marker='s', linewidth=2, label='Rain', color='darkblue')
ax1.set_xlabel('Hour of Day', fontsize=12)
ax1.set_ylabel('Average Order Count', fontsize=12)
ax1.set_title('Weekdays: Rain Impact', fontsize=14, fontweight='bold')
ax1.set_xticks(range(0, 24))
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Weekends: Rain vs No Rain
ax2.plot(weekend_no_rain['hour'], weekend_no_rain['avg_order_count'],
         marker='o', linewidth=2, label='No Rain', color='coral')
ax2.plot(weekend_rain['hour'], weekend_rain['avg_order_count'],
         marker='s', linewidth=2, label='Rain', color='darkred')
ax2.set_xlabel('Hour of Day', fontsize=12)
ax2.set_ylabel('Average Order Count', fontsize=12)
ax2.set_title('Weekends: Rain Impact', fontsize=14, fontweight='bold')
ax2.set_xticks(range(0, 24))
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
