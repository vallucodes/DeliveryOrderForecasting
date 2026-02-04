import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Load the data
df = pd.read_csv('orders_spring_2022.csv')

# Convert the timestamp string to datetime
df['order_placed_at_utc'] = pd.to_datetime(df['order_placed_at_utc'])

# Extract time and precipitation features
df['date'] = df['order_placed_at_utc'].dt.date
df['hour'] = df['order_placed_at_utc'].dt.hour
df['day_of_week'] = df['order_placed_at_utc'].dt.dayofweek      # 0 = Monday, 6 = Sunday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)   # Saturday, Sunday
df['has_rain'] = (df['precipitation'] > 0.4).astype(int)

# Group data together for each hour
hourly = df.groupby(['date', 'hour']).agg({
    'order_placed_at_utc': 'count',             # This counts orders per hour
    'precipitation': 'first',                   # Assuming same precipitation for the whole hour
    'courier_supply_index': 'mean',             # Average courier supply in that hour
    'day_of_week': 'first',                     # Keep day of week
    'is_weekend': 'first',                      # Weekend flag
    'has_rain': 'first'                         # Rain forecast
}).reset_index()

# Rename the count column
hourly.rename(columns={'order_placed_at_utc': 'order_count'}, inplace=True)






# Explore weekdays vs weekend hourly orders. Separate if rain forecast for the hour >0.

# Create 4 different groups
weekday_no_rain =   hourly[(hourly['day_of_week'] < 5) & (hourly['has_rain'] == 0)].groupby('hour')['order_count'].mean().reset_index()
weekday_rain =      hourly[(hourly['day_of_week'] < 5) & (hourly['has_rain'] == 1)].groupby('hour')['order_count'].mean().reset_index()
weekend_no_rain =   hourly[(hourly['day_of_week'] >= 5) & (hourly['has_rain'] == 0)].groupby('hour')['order_count'].mean().reset_index()
weekend_rain =      hourly[(hourly['day_of_week'] >= 5) & (hourly['has_rain'] == 1)].groupby('hour')['order_count'].mean().reset_index()

# Rename order_count columns
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





# # Prepare Labels and the Red-to-Green Gradient
# days_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# # Generate 7 colors from the RdYlGn colormap (0.0 is Red, 1.0 is Green)
# colors_gradient = [plt.cm.RdYlGn(i/6) for i in range(7)]

# # Total Orders per Weekday
# weekday_totals = hourly.groupby('day_of_week')['order_count'].sum().reset_index()

# plt.figure(figsize=(10, 6))
# plt.bar(days_names, [weekday_totals.loc[weekday_totals['day_of_week'] == i, 'order_count'].values[0] for i in range(7)],
#         color=colors_gradient)

# plt.title('Total Order Count (Monday to Sunday Gradient)', fontsize=14)
# plt.xlabel('Day of the Week')
# plt.ylabel('Total Orders')
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.show()

# # Hourly Patterns by Weekday
# avg_hourly_weekday = hourly.groupby(['day_of_week', 'hour'])['order_count'].mean().reset_index()

# plt.figure(figsize=(12, 7))
# for i, day in enumerate(days_names):
#     day_subset = avg_hourly_weekday[avg_hourly_weekday['day_of_week'] == i]
#     plt.plot(day_subset['hour'], day_subset['order_count'],
#              label=day, color=colors_gradient[i], linewidth=2.5)

# plt.title('Average Hourly Order Trends (Red to Green)', fontsize=14)
# plt.xlabel('Hour of Day (UTC)')
# plt.ylabel('Average Order Count')
# plt.xticks(range(24))
# plt.legend(title='Weekday', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, linestyle=':', alpha=0.5)
# plt.tight_layout()
# plt.show()






# # Explore how different precipitation thresholds affect orders amount

# # Define precipitation thresholds
# precip_thresholds = [0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# # Use a colormap from light to dark
# colors = plt.cm.coolwarm(np.linspace(0, 1, len(precip_thresholds)))

# plt.figure(figsize=(10,6))

# for p, c in zip(precip_thresholds, colors):
#     filtered = hourly[hourly['precipitation'] > p].groupby('hour')['order_count'].mean().reset_index()
#     plt.plot(filtered['hour'], filtered['order_count'], label=f'precip > {p}', color=c)

# plt.xlabel('Hour of Day')
# plt.ylabel('Average Orders')
# plt.title('Hourly Average Orders by Precipitation Level')
# plt.legend()
# plt.grid(True)
# plt.show()







# # Explore food delivery vs retail orders

# # Separate order categories
# categories = df['order_category'].unique()

# plt.figure(figsize=(12,6))

# # Optional: colors for each category
# colors = ['steelblue', 'coral']

# for cat, c in zip(categories, colors):
#     # Filter category
#     cat_data = df[df['order_category'] == cat]

#     # Aggregate hourly
#     hourly_cat = cat_data.groupby(['date', 'hour']).agg({
#         'order_placed_at_utc': 'count'
#     }).reset_index()

#     hourly_cat.rename(columns={'order_placed_at_utc': 'order_count'}, inplace=True)

#     # Average per hour across all days
#     avg_hourly = hourly_cat.groupby('hour')['order_count'].mean().reset_index()

#     # Plot
#     plt.plot(avg_hourly['hour'], avg_hourly['order_count'],
#              marker='o', linewidth=2, label=cat, color=c)

# plt.xlabel('Hour of Day')
# plt.ylabel('Average Orders')
# plt.title('Hourly Average Orders by Order Category')
# plt.xticks(range(0, 24))
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()






# # Explore how easter holidays affected

# # Force convert date to datetime
# hourly['date'] = pd.to_datetime(hourly['date'])

# # Define holiday dates separately
# weekday_holidays = pd.to_datetime(['2022-04-15', '2022-04-18'])
# weekend_holidays = pd.to_datetime(['2022-04-16', '2022-04-17'])

# # Create day_type with 4 categories
# hourly['day_type'] = np.where(
#     hourly['date'].isin(weekday_holidays), 'Weekday Holiday',
#     np.where(hourly['date'].isin(weekend_holidays), 'Weekend Holiday',
#     np.where(hourly['day_of_week'] < 5, 'Weekday', 'Weekend')))

# # Aggregate average orders per hour for each day_type
# avg_hourly_by_daytype = hourly.groupby(['day_type', 'hour'])['order_count'].mean().reset_index()

# # Plot
# plt.figure(figsize=(12, 6))
# day_types = ['Weekday', 'Weekend', 'Weekday Holiday', 'Weekend Holiday']
# colors = ['steelblue', 'coral', 'green', 'darkgreen']
# for dt, c in zip(day_types, colors):
#     subset = avg_hourly_by_daytype[avg_hourly_by_daytype['day_type'] == dt]
#     plt.plot(subset['hour'], subset['order_count'], marker='o', linewidth=2, label=dt, color=c)
# plt.xlabel('Hour of Day')
# plt.ylabel('Average Orders')
# plt.title('Hourly Average Orders: Weekday vs Weekend vs Holidays')
# plt.xticks(range(0, 24))
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()





# # Check if poisson regression is suitable

# from scipy import stats

# # Check ratio for each group separately
# groups = {
#     'Weekday': hourly[hourly['is_weekend'] == 0],
#     'Weekend': hourly[hourly['is_weekend'] == 1]
# }

# print("Variance/Mean ratio by group:")
# print("-" * 45)
# for name, group in groups.items():
#     mean = group['order_count'].mean()
#     var = group['order_count'].var()
#     ratio = var / mean
#     print(f"{name:<12} Mean: {mean:<8.2f} Var: {var:<10.2f} Ratio: {ratio:.2f}")

# # Even more detailed - check per hour per day type
# print("\n\nRatio per hour:")
# print("-" * 55)
# print(f"{'Hour':<6} {'Weekday Ratio':<18} {'Weekend Ratio':<18}")
# print("-" * 55)

# for hour in range(24):
#     wd = hourly[(hourly['hour'] == hour) & (hourly['is_weekend'] == 0)]['order_count']
#     we = hourly[(hourly['hour'] == hour) & (hourly['is_weekend'] == 1)]['order_count']

#     wd_ratio = wd.var() / wd.mean() if wd.mean() > 0 else 0
#     we_ratio = we.var() / we.mean() if we.mean() > 0 else 0

#     print(f"{hour:<6} {wd_ratio:<18.2f} {we_ratio:<18.2f}")







# # --- DATA EXPLORATION: TIME TREND ANALYSIS ---

# # 1. Aggregate to Daily level to see the big picture
# daily_orders = hourly.groupby('date')['order_count'].sum().reset_index()
# daily_orders['date'] = pd.to_datetime(daily_orders['date'])

# # 2. Calculate a 7-Day Rolling Average
# daily_orders['7_day_avg'] = daily_orders['order_count'].rolling(window=7).mean()

# # 3. Calculate a Linear Trend Line
# # Convert dates to simple numbers (Day 0, Day 1, Day 2...) for regression
# x_values = np.arange(len(daily_orders))
# y_values = daily_orders['order_count'].values

# # Fit a line
# slope, intercept = np.polyfit(x_values, y_values, 1)
# trend_line = slope * x_values + intercept

# print(f"Trend Slope: {slope:.4f} orders/day")
# if abs(slope) < 0.1:
#     print("CONCLUSION: The trend is effectively FLAT.")
# elif slope > 0:
#     print("CONCLUSION: There is a POSITIVE growth trend.")
# else:
#     print("CONCLUSION: There is a NEGATIVE declining trend.")

# # --- PLOTTING ---
# plt.figure(figsize=(12, 6))

# # Plot raw daily data as faint dots
# plt.scatter(daily_orders['date'], daily_orders['order_count'],
#             color='gray', alpha=0.5, label='Daily Total Orders')

# # Plot the 7-day average (The Smooth Trend)
# plt.plot(daily_orders['date'], daily_orders['7_day_avg'],
#          color='blue', linewidth=3, label='7-Day Rolling Average')

# # Plot the mathematical trend line
# plt.plot(daily_orders['date'], trend_line,
#          color='red', linestyle='--', linewidth=2, label=f'Linear Trend (Slope={slope:.2f})')

# plt.title('Order Volume Over Time (Spring 2022)', fontsize=14)
# plt.xlabel('Date')
# plt.ylabel('Total Orders')
# plt.legend()
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.tight_layout()
# plt.show()

# # 1. Prepare data for statistics
# x_values = np.arange(len(daily_orders))
# y_values = daily_orders['order_count'].values

# # 2. Calculate statistics using linregress
# # This returns:
# # slope: How much orders change per day
# # intercept: The starting point
# # r_value: Correlation coefficient
# # p_value: Statistical significance of the slope
# # std_err: Standard error of the estimate
# slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)

# # 3. Descriptive Stats
# stats_desc = daily_orders['order_count'].describe()

# # --- PRINT STATISTICAL REPORT ---
# print("-" * 40)
# print("STATISTICAL TREND ANALYSIS REPORT")
# print("-" * 40)
# print(f"Data Range: {daily_orders['date'].min()} to {daily_orders['date'].max()}")
# print(f"Total Days: {len(daily_orders)}")

# print(f"\n1. DESCRIPTIVE STATS (Daily Orders):")
# print(f"   Mean:   {stats_desc['mean']:.2f}")
# print(f"   StdDev: {stats_desc['std']:.2f}")
# print(f"   Min:    {stats_desc['min']:.0f}")
# print(f"   Max:    {stats_desc['max']:.0f}")

# print(f"\n2. TREND ANALYSIS:")
# print(f"   Slope (m):        {slope:.4f} (orders gained/lost per day)")
# print(f"   Intercept (c):    {intercept:.2f}")
# print(f"   Correlation (r):  {r_value:.4f}")
# print(f"   R-squared (r^2):  {r_value**2:.4f} (Variance explained by time)")
# print(f"   Std Error:        {std_err:.4f}")

# print(f"\n3. SIGNIFICANCE TEST:")
# print(f"   P-value:          {p_value:.5f}")

# print("-" * 40)
# print("CONCLUSION:")
# if p_value > 0.05:
#     print("   The trend is NOT statistically significant (p > 0.05).")
#     print("   We cannot reject the null hypothesis that the slope is zero.")
#     print("   -> CONFIRMED: No meaningful time trend exists.")
# else:
#     print("   The trend IS statistically significant (p < 0.05).")
#     if slope > 0:
#         print("   -> There is a verified POSITIVE growth trend.")
#     else:
#         print("   -> There is a verified NEGATIVE declining trend.")
# print("-" * 40)
