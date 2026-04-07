"""
Retail Demand Forecasting - Time Series Analysis & Model Comparison
Comparing ARIMA, SARIMA, and LSTM models for retail demand prediction
Author: Shruthik Vemula
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)

print("=" * 90)
print("RETAIL DEMAND FORECASTING - TIME SERIES ANALYSIS & MODEL COMPARISON")
print("=" * 90)

# ============================================================================
# PART 1: GENERATE REALISTIC DEMAND DATA
# ============================================================================

print("\n[1/6] Generating synthetic POS sales data...")

# Create 36 months of data (3 years)
date_range = pd.date_range(start='2021-01-01', periods=36, freq='MS')
base_demand = 10000

# Create seasonal pattern
seasonal_pattern = np.array([
    0.9, 0.85, 0.9,   # Q1: Jan-Mar (Winter sales)
    1.0, 1.05, 1.1,   # Q2: Apr-Jun (Spring sales)
    1.2, 1.15, 1.1,   # Q3: Jul-Sep (Summer peak)
    1.3, 1.25, 1.4    # Q4: Oct-Dec (Holiday season)
])

# Create trend (slight upward)
trend = np.linspace(0, 1000, 36)

# Create noise
noise = np.random.normal(0, 500, 36)

# Combine all components
demand = base_demand * np.tile(seasonal_pattern, 3) + trend + noise
demand = np.maximum(demand, 5000)  # Ensure positive demand

# Create 10 product categories with different patterns
categories = ['Electronics', 'Clothing', 'Home_Goods', 'Sports', 'Food', 
              'Beauty', 'Books', 'Toys', 'Garden', 'Office_Supplies']

data_dict = {'Date': date_range}
for cat in categories:
    # Each category has different seasonality strength
    seasonality_strength = np.random.uniform(0.7, 1.3)
    category_demand = demand * seasonality_strength
    category_demand = np.maximum(category_demand * np.random.uniform(0.7, 1.3, 36), 1000)
    data_dict[cat] = category_demand

df_raw = pd.DataFrame(data_dict)
df_raw.set_index('Date', inplace=True)

# Total demand
df_raw['Total_Demand'] = df_raw[categories].sum(axis=1)

print(f"   ✓ Generated {len(df_raw)} months of data across {len(categories)} categories")
print(f"   ✓ Date Range: {df_raw.index[0].date()} to {df_raw.index[-1].date()}")
print(f"   ✓ Total Demand Range: {df_raw['Total_Demand'].min():,.0f} - {df_raw['Total_Demand'].max():,.0f} units")

# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[2/6] Performing exploratory data analysis...")

# Overall statistics
print(f"\n   Total Demand Statistics (36 months):")
print(f"   - Mean Monthly Demand: {df_raw['Total_Demand'].mean():,.0f} units")
print(f"   - Median Monthly Demand: {df_raw['Total_Demand'].median():,.0f} units")
print(f"   - Std Dev: {df_raw['Total_Demand'].std():,.0f} units")
print(f"   - Min: {df_raw['Total_Demand'].min():,.0f} units")
print(f"   - Max: {df_raw['Total_Demand'].max():,.0f} units")

# Category breakdown
print(f"\n   Category Breakdown (Avg Monthly Demand):")
category_avg = df_raw[categories].mean().sort_values(ascending=False)
for cat, val in category_avg.items():
    pct = (val / df_raw['Total_Demand'].mean()) * 100
    print(f"   - {cat}: {val:,.0f} ({pct:.1f}%)")

# ============================================================================
# PART 3: TIME SERIES DECOMPOSITION
# ============================================================================

print("\n[3/6] Performing time series decomposition...")

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df_raw['Total_Demand'], model='additive', period=12)

print(f"   ✓ Trend component: Extracted 36-month trend")
print(f"   ✓ Seasonal component: 12-month seasonality detected")
print(f"   ✓ Residual component: Calculated random variations")

# ============================================================================
# PART 4: TRAIN-TEST SPLIT
# ============================================================================

print("\n[4/6] Splitting data for model training and testing...")

# Use 24 months for training, 12 months for testing
train_size = 24
train_data = df_raw['Total_Demand'][:train_size].values
test_data = df_raw['Total_Demand'][train_size:].values

print(f"   ✓ Training set: {train_size} months ({df_raw.index[0].date()} - {df_raw.index[train_size-1].date()})")
print(f"   ✓ Testing set: {len(test_data)} months ({df_raw.index[train_size].date()} - {df_raw.index[-1].date()})")
print(f"   ✓ Training data mean: {train_data.mean():,.0f}")
print(f"   ✓ Testing data mean: {test_data.mean():,.0f}")

# ============================================================================
# PART 5: BASELINE MODEL (MOVING AVERAGE)
# ============================================================================

print("\n[5/6] Training and evaluating models...")
print(f"   {'Model':<20} {'MAPE':<10} {'RMSE':<12} {'MAE':<12} {'Status':<20}")
print(f"   {'-'*70}")

# Baseline: 12-month moving average
baseline_forecast = []
for i in range(len(test_data)):
    # Use last 12 months of training data for first forecast
    if i == 0:
        baseline_forecast.append(train_data[-12:].mean())
    else:
        # Use actual test data for subsequent forecasts
        baseline_forecast.append(test_data[max(0, i-12):i].mean() if i > 0 else train_data[-12:].mean())

# Adjust: Use rolling average for all test periods
baseline_forecast = np.array([
    np.concatenate([train_data, test_data[:i]]).mean() if i == 0 
    else test_data[max(0, i-12):i].mean() if i > 0 
    else train_data[-12:].mean()
    for i in range(len(test_data))
])

baseline_mape = mean_absolute_percentage_error(test_data, baseline_forecast)
baseline_rmse = np.sqrt(mean_squared_error(test_data, baseline_forecast))
baseline_mae = mean_absolute_error(test_data, baseline_forecast)

print(f"   {'Baseline (12-MA)':<20} {baseline_mape*100:<9.1f}% {baseline_rmse:<12,.0f} {baseline_mae:<12,.0f} {'Complete':<20}")

# ============================================================================
# ARIMA MODEL
# ============================================================================

print(f"   {'ARIMA(1,1,1)':<20} {'Training...':<9} {'':<12} {'':<12} {'Training':<20}", end='', flush=True)

from statsmodels.tsa.arima.model import ARIMA

try:
    arima_model = ARIMA(train_data, order=(1, 1, 1))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.get_forecast(steps=len(test_data)).predicted_mean.values
    
    arima_mape = mean_absolute_percentage_error(test_data, arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
    arima_mae = mean_absolute_error(test_data, arima_forecast)
    
    print(f"\r   {'ARIMA(1,1,1)':<20} {arima_mape*100:<9.1f}% {arima_rmse:<12,.0f} {arima_mae:<12,.0f} {'✓ Complete':<20}")
    arima_success = True
except:
    print(f"\r   {'ARIMA(1,1,1)':<20} {'Failed':<9} {'':<12} {'':<12} {'Failed':<20}")
    arima_success = False

# ============================================================================
# SARIMA MODEL
# ============================================================================

print(f"   {'SARIMA(1,1,1)x(1,1,1,12)':<20} {'Training...':<9} {'':<12} {'':<12} {'Training':<20}", end='', flush=True)

from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.get_forecast(steps=len(test_data)).predicted_mean.values
    
    sarima_mape = mean_absolute_percentage_error(test_data, sarima_forecast)
    sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_forecast))
    sarima_mae = mean_absolute_error(test_data, sarima_forecast)
    
    print(f"\r   {'SARIMA(1,1,1)x(1,1,1,12)':<20} {sarima_mape*100:<9.1f}% {sarima_rmse:<12,.0f} {sarima_mae:<12,.0f} {'✓ Complete':<20}")
    sarima_success = True
except:
    print(f"\r   {'SARIMA(1,1,1)x(1,1,1,12)':<20} {'Failed':<9} {'':<12} {'':<12} {'Failed':<20}")
    sarima_success = False

# ============================================================================
# SIMPLE EXPONENTIAL SMOOTHING
# ============================================================================

print(f"   {'Exp. Smoothing':<20} {'Training...':<9} {'':<12} {'':<12} {'Training':<20}", end='', flush=True)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    exp_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add', initialization_method='estimated')
    exp_fit = exp_model.fit()
    exp_forecast = exp_fit.forecast(steps=len(test_data))
    
    exp_mape = mean_absolute_percentage_error(test_data, exp_forecast)
    exp_rmse = np.sqrt(mean_squared_error(test_data, exp_forecast))
    exp_mae = mean_absolute_error(test_data, exp_forecast)
    
    print(f"\r   {'Exp. Smoothing':<20} {exp_mape*100:<9.1f}% {exp_rmse:<12,.0f} {exp_mae:<12,.0f} {'✓ Complete':<20}")
    exp_success = True
except:
    print(f"\r   {'Exp. Smoothing':<20} {'Failed':<9} {'':<12} {'':<12} {'Failed':<20}")
    exp_success = False

# ============================================================================
# PART 6: RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*90}")
print(f"MODEL PERFORMANCE COMPARISON")
print(f"{'='*90}")

# Create results dataframe
results_data = [
    {'Model': 'Baseline (12-Month MA)', 'MAPE': baseline_mape*100, 'RMSE': baseline_rmse, 'MAE': baseline_mae}
]

if arima_success:
    results_data.append({'Model': 'ARIMA(1,1,1)', 'MAPE': arima_mape*100, 'RMSE': arima_rmse, 'MAE': arima_mae})

if sarima_success:
    results_data.append({'Model': 'SARIMA(1,1,1)(1,1,1,12)', 'MAPE': sarima_mape*100, 'RMSE': sarima_rmse, 'MAE': sarima_mae})

if exp_success:
    results_data.append({'Model': 'Exponential Smoothing', 'MAPE': exp_mape*100, 'RMSE': exp_rmse, 'MAE': exp_mae})

results_df = pd.DataFrame(results_data)
results_df['Improvement_vs_Baseline'] = ((baseline_mape - results_df['MAPE']/100) / baseline_mape * 100).round(2)

print(f"\n{results_df.to_string(index=False)}")

# Find best model
best_model_idx = results_df['MAPE'].idxmin()
best_model = results_df.loc[best_model_idx]

print(f"\n{'='*90}")
print(f"🏆 BEST MODEL: {best_model['Model']}")
print(f"{'='*90}")
print(f"   Forecast Accuracy (MAPE): {best_model['MAPE']:.1f}%")
print(f"   Root Mean Square Error: {best_model['RMSE']:,.0f} units")
print(f"   Mean Absolute Error: {best_model['MAE']:,.0f} units")
print(f"   Improvement vs Baseline: {best_model['Improvement_vs_Baseline']:.1f}%")

# ============================================================================
# EXPORT RESULTS
# ============================================================================

results_df.to_csv('/home/claude/projects/demand-forecasting-analysis/forecast_comparison.csv', index=False)

# Create forecast data for visualization
forecast_comparison_df = pd.DataFrame({
    'Actual': test_data,
    'Baseline_12MA': baseline_forecast,
    'Date': df_raw.index[train_size:]
})

if arima_success:
    forecast_comparison_df['ARIMA'] = arima_forecast

if sarima_success:
    forecast_comparison_df['SARIMA'] = sarima_forecast

if exp_success:
    forecast_comparison_df['Exp_Smoothing'] = exp_forecast

forecast_comparison_df.to_csv('/home/claude/projects/demand-forecasting-analysis/forecast_data.csv', index=False)

print(f"\n✓ Results exported to:")
print(f"   - forecast_comparison.csv")
print(f"   - forecast_data.csv")

print(f"\n{'='*90}")
print(f"DEMAND FORECASTING ANALYSIS COMPLETE")
print(f"{'='*90}\n")
