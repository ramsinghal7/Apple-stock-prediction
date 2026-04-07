import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# pip install statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE  = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'
PLOTS = os.path.join(BASE, 'outputs', 'plots')
os.makedirs(PLOTS, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0a0e1a',
    'axes.facecolor':    '#111827',
    'axes.edgecolor':    '#1e2d45',
    'axes.labelcolor':   '#64748b',
    'text.color':        '#e2e8f0',
    'xtick.color':       '#64748b',
    'ytick.color':       '#64748b',
    'grid.color':        '#1e2d45',
    'grid.alpha':        0.5,
    'lines.linewidth':   1.5,
    'font.family':       'monospace',
})

print("=" * 60)
print("   ARIMA MODEL — AAPL Stock Price Prediction")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading data...")

df = pd.read_csv(os.path.join(BASE, 'data', 'AAPL.csv'))
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)

# Use only Close price — ARIMA is univariate (one variable at a time)
close = df['Close'].copy()

print(f"  Total data points : {len(close)}")
print(f"  Date range        : {close.index[0].date()} -> {close.index[-1].date()}")
print(f"  Min price         : ${close.min():.2f}")
print(f"  Max price         : ${close.max():.2f}")

# ── Use RECENT 3 years for ARIMA (ARIMA is slow on 40 years of data)
# This is intentional — ARIMA works best on recent, stable data
recent = close[-756:].copy()   # ~3 years of trading days
print(f"\n  Using last 3 years: {recent.index[0].date()} -> {recent.index[-1].date()}")
print(f"  Recent data points: {len(recent)}")

# ══════════════════════════════════════════════════════════════════
# STEP 2: STATIONARITY TEST (ADF Test)
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 2] Checking Stationarity (ADF Test)...")
print("-" * 40)

def adf_test(series, name="Series"):
    result = adfuller(series.dropna())
    print(f"\n  {name}")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Critical 5%   : {result[4]['5%']:.4f}")
    if result[1] < 0.05:
        print(f"  Result        : STATIONARY (p < 0.05) -- OK to use")
        return True
    else:
        print(f"  Result        : NOT STATIONARY -- need differencing")
        return False

is_stationary = adf_test(recent, "Original Close Price")

# Apply first-order differencing if not stationary
diff1 = recent.diff().dropna()
print()
is_stationary_d1 = adf_test(diff1, "After 1st Differencing (d=1)")

# Apply second differencing if still not stationary
diff2 = diff1.diff().dropna()
print()
is_stationary_d2 = adf_test(diff2, "After 2nd Differencing (d=2)")

# Choose d value
if is_stationary:
    d = 0
    print("\n  ==> Using d=0 (no differencing needed)")
elif is_stationary_d1:
    d = 1
    print("\n  ==> Using d=1 (first-order differencing)")
else:
    d = 2
    print("\n  ==> Using d=2 (second-order differencing)")

# ══════════════════════════════════════════════════════════════════
# STEP 3: ACF & PACF PLOTS (to choose p and q)
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 3] Plotting ACF and PACF to find p and q...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('ARIMA Parameter Selection — ACF & PACF', fontsize=14, fontweight='bold', color='#e2e8f0')

# Original series
axes[0, 0].plot(recent.index, recent.values, color='#00d4ff', linewidth=0.8)
axes[0, 0].set_title('Original Close Price (3 Years)', color='#e2e8f0')
axes[0, 0].set_xlabel('Date'); axes[0, 0].set_ylabel('Price (USD)')

# Differenced series
axes[0, 1].plot(diff1.index, diff1.values, color='#f59e0b', linewidth=0.8)
axes[0, 1].axhline(0, color='#ef4444', linestyle='--', linewidth=1)
axes[0, 1].set_title('1st Differenced Series', color='#e2e8f0')
axes[0, 1].set_xlabel('Date'); axes[0, 1].set_ylabel('Price Change')

# ACF plot — tells us q (MA order)
plot_acf(diff1, lags=30, ax=axes[1, 0], color='#7c3aed',
         title='ACF (tells q — MA order)', alpha=0.05)
axes[1, 0].set_title('ACF Plot — q (MA order)', color='#e2e8f0')

# PACF plot — tells us p (AR order)
plot_pacf(diff1, lags=30, ax=axes[1, 1], color='#10b981',
          title='PACF (tells p — AR order)', alpha=0.05, method='ywm')
axes[1, 1].set_title('PACF Plot — p (AR order)', color='#e2e8f0')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'arima_01_acf_pacf.png'), dpi=150)
plt.show()
print("  Saved: arima_01_acf_pacf.png")

# ══════════════════════════════════════════════════════════════════
# STEP 4: AUTO-FIND BEST (p, d, q) PARAMETERS
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 4] Finding Best ARIMA Parameters (p, d, q)...")
print("  Testing combinations of p=[0-4], d=[0-2], q=[0-4]")
print("  Using AIC (Akaike Info Criterion) — lower is better")
print("-" * 40)

best_aic = np.inf
best_order = (1, 1, 1)
results_grid = []

# Grid search (small range for speed)
for p in range(0, 5):
    for q in range(0, 5):
        try:
            model = ARIMA(recent, order=(p, d, q))
            fitted = model.fit()
            aic = fitted.aic
            results_grid.append({'p': p, 'd': d, 'q': q, 'AIC': round(aic, 2)})
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
        except:
            pass

# Show top 5 results
grid_df = pd.DataFrame(results_grid).sort_values('AIC').head(10)
print("\n  Top 10 ARIMA combinations by AIC:")
print(grid_df.to_string(index=False))
print(f"\n  ==> BEST ORDER: ARIMA{best_order}  (AIC={best_aic:.2f})")

# ══════════════════════════════════════════════════════════════════
# STEP 5: TRAIN-TEST SPLIT
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 5] Train-Test Split (80/20, time-based)...")

split = int(len(recent) * 0.80)
train = recent.iloc[:split]
test  = recent.iloc[split:]

print(f"  Train: {len(train)} points  ({train.index[0].date()} -> {train.index[-1].date()})")
print(f"  Test : {len(test)} points   ({test.index[0].date()} -> {test.index[-1].date()})")

# ══════════════════════════════════════════════════════════════════
# STEP 6: FIT BEST ARIMA MODEL
# ══════════════════════════════════════════════════════════════════
print(f"\n[STEP 6] Fitting ARIMA{best_order} on training data...")

model = ARIMA(train, order=best_order)
fitted_model = model.fit()

print("\n  === ARIMA Model Summary ===")
print(fitted_model.summary())

# ══════════════════════════════════════════════════════════════════
# STEP 7: FORECASTING
# ══════════════════════════════════════════════════════════════════
print(f"\n[STEP 7] Forecasting {len(test)} steps ahead...")

# METHOD A: One-shot forecast (faster, less accurate)
forecast_result = fitted_model.get_forecast(steps=len(test))
forecast_mean   = forecast_result.predicted_mean
conf_int        = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval

# METHOD B: Rolling (walk-forward) forecast — more realistic
# Re-trains model each day using all data up to that point
print("  Running walk-forward forecast (this is more realistic)...")
rolling_preds = []
history = list(train)

for i in range(len(test)):
    try:
        m = ARIMA(history, order=best_order)
        f = m.fit()
        pred = f.forecast(steps=1)[0]
    except:
        pred = history[-1]   # fallback: use last known value
    rolling_preds.append(pred)
    history.append(test.iloc[i])   # add actual value for next iteration
    if (i+1) % 50 == 0:
        print(f"    Progress: {i+1}/{len(test)} steps done...")

rolling_preds = np.array(rolling_preds)
print("  Walk-forward forecast complete!")

# ══════════════════════════════════════════════════════════════════
# STEP 8: EVALUATE
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 8] Model Evaluation...")
print("=" * 50)

actual = test.values

# One-shot metrics
mae_os   = mean_absolute_error(actual, forecast_mean)
rmse_os  = np.sqrt(mean_squared_error(actual, forecast_mean))
mape_os  = np.mean(np.abs((actual - forecast_mean) / actual)) * 100

# Rolling metrics
mae_roll  = mean_absolute_error(actual, rolling_preds)
rmse_roll = np.sqrt(mean_squared_error(actual, rolling_preds))
mape_roll = np.mean(np.abs((actual - rolling_preds) / actual)) * 100

print(f"\n  ARIMA{best_order} — One-Shot Forecast:")
print(f"  MAE  : {mae_os:.4f}  (avg error in USD)")
print(f"  RMSE : {rmse_os:.4f}  (penalizes big errors)")
print(f"  MAPE : {mape_os:.2f}%  (mean absolute % error)")

print(f"\n  ARIMA{best_order} — Walk-Forward (Rolling) Forecast:")
print(f"  MAE  : {mae_roll:.4f}  (avg error in USD)")
print(f"  RMSE : {rmse_roll:.4f}  (penalizes big errors)")
print(f"  MAPE : {mape_roll:.2f}%  (mean absolute % error)")

# Compare with baseline (naive: predict yesterday's price)
naive_pred = actual[:-1]
mae_naive  = mean_absolute_error(actual[1:], naive_pred)
print(f"\n  Naive Baseline (yesterday's price as prediction):")
print(f"  MAE  : {mae_naive:.4f}")
print(f"\n  ==> ARIMA improvement over Naive: {((mae_naive - mae_roll)/mae_naive)*100:.1f}%")

# ══════════════════════════════════════════════════════════════════
# STEP 9: PLOTS
# ══════════════════════════════════════════════════════════════════
print("\n[STEP 9] Generating plots...")

# ── Plot 1: Full picture with train, test, forecast
fig, ax = plt.subplots(figsize=(16, 6))
fig.suptitle(f'ARIMA{best_order} — Apple Stock Price Forecast', fontsize=14, fontweight='bold', color='#e2e8f0')

ax.plot(train.index, train.values, color='#00d4ff', linewidth=0.9, label='Training Data', alpha=0.8)
ax.plot(test.index,  actual,       color='#10b981', linewidth=1.2, label='Actual (Test)')
ax.plot(test.index,  forecast_mean, color='#ef4444', linewidth=1.2, linestyle='--', label='One-Shot Forecast')
ax.plot(test.index,  rolling_preds, color='#f59e0b', linewidth=1.2, linestyle=':',  label='Walk-Forward Forecast')

# Confidence interval shading
ax.fill_between(test.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color='#ef4444', alpha=0.1, label='95% Confidence Interval')

ax.axvline(test.index[0], color='#7c3aed', linewidth=1.5, linestyle='--', label='Train/Test Split')
ax.set_xlabel('Date'); ax.set_ylabel('Price (USD)')
ax.legend(facecolor='#1e2d45', labelcolor='#e2e8f0', fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'arima_02_full_forecast.png'), dpi=150)
plt.show()
print("  Saved: arima_02_full_forecast.png")

# ── Plot 2: Zoom in on test period
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('ARIMA Forecast — Test Period Zoom', fontsize=13, fontweight='bold', color='#e2e8f0')

# One-shot
axes[0].plot(test.index, actual,        color='#10b981', linewidth=1.5, label='Actual')
axes[0].plot(test.index, forecast_mean, color='#ef4444', linewidth=1.2, linestyle='--', label='Forecast')
axes[0].fill_between(test.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='#ef4444', alpha=0.15)
axes[0].set_title('One-Shot Forecast vs Actual', color='#e2e8f0')
axes[0].set_xlabel('Date'); axes[0].set_ylabel('Price (USD)')
axes[0].legend(facecolor='#1e2d45', labelcolor='#e2e8f0', fontsize=9)
axes[0].grid(True, alpha=0.3)

# Rolling
axes[1].plot(test.index, actual,        color='#10b981', linewidth=1.5, label='Actual')
axes[1].plot(test.index, rolling_preds, color='#f59e0b', linewidth=1.2, linestyle='--', label='Walk-Forward Forecast')
axes[1].set_title('Walk-Forward Forecast vs Actual', color='#e2e8f0')
axes[1].set_xlabel('Date'); axes[1].set_ylabel('Price (USD)')
axes[1].legend(facecolor='#1e2d45', labelcolor='#e2e8f0', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'arima_03_test_zoom.png'), dpi=150)
plt.show()
print("  Saved: arima_03_test_zoom.png")

# ── Plot 3: Residuals analysis
residuals = fitted_model.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('ARIMA Residuals Analysis', fontsize=13, fontweight='bold', color='#e2e8f0')

# Residuals over time
axes[0, 0].plot(residuals.index, residuals.values, color='#7c3aed', linewidth=0.8)
axes[0, 0].axhline(0, color='#ef4444', linestyle='--', linewidth=1)
axes[0, 0].set_title('Residuals Over Time', color='#e2e8f0')
axes[0, 0].set_xlabel('Date'); axes[0, 0].set_ylabel('Residual')

# Residuals histogram
axes[0, 1].hist(residuals, bins=50, color='#7c3aed', alpha=0.7, edgecolor='none')
axes[0, 1].set_title('Residuals Distribution (should be ~normal)', color='#e2e8f0')
axes[0, 1].set_xlabel('Residual Value'); axes[0, 1].set_ylabel('Frequency')

# ACF of residuals (should have no pattern)
plot_acf(residuals, lags=30, ax=axes[1, 0], color='#00d4ff', alpha=0.05)
axes[1, 0].set_title('ACF of Residuals (should be ~zero)', color='#e2e8f0')

# Actual error on test set
error = actual - rolling_preds
axes[1, 1].plot(test.index, error, color='#f59e0b', linewidth=0.8)
axes[1, 1].axhline(0, color='#ef4444', linestyle='--', linewidth=1)
axes[1, 1].fill_between(test.index, error, alpha=0.2, color='#f59e0b')
axes[1, 1].set_title('Forecast Error on Test Set', color='#e2e8f0')
axes[1, 1].set_xlabel('Date'); axes[1, 1].set_ylabel('Error (USD)')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'arima_04_residuals.png'), dpi=150)
plt.show()
print("  Saved: arima_04_residuals.png")

# ── Plot 4: Future 30-day forecast
print("\n[STEP 10] Forecasting next 30 trading days into the future...")

future_model  = ARIMA(recent, order=best_order)
future_fitted = future_model.fit()
future_fc     = future_fitted.get_forecast(steps=30)
future_mean   = future_fc.predicted_mean
future_ci     = future_fc.conf_int(alpha=0.05)

# Create future date index
last_date    = recent.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
future_mean.index = future_dates
future_ci.index   = future_dates

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('ARIMA — Next 30 Trading Days Forecast', fontsize=14, fontweight='bold', color='#e2e8f0')

# Show last 100 days of actual + 30 future
ax.plot(recent.index[-100:], recent.values[-100:], color='#00d4ff', linewidth=1.5, label='Recent Actual Price')
ax.plot(future_mean.index,   future_mean.values,   color='#f59e0b', linewidth=2, linestyle='--', label='30-Day Forecast')
ax.fill_between(future_dates, future_ci.iloc[:,0], future_ci.iloc[:,1],
                color='#f59e0b', alpha=0.15, label='95% Confidence Band')
ax.axvline(recent.index[-1], color='#7c3aed', linewidth=1.5, linestyle='--', label='Today')

# Annotate last actual and first/last forecast
ax.annotate(f'Last: ${recent.values[-1]:.2f}',
            xy=(recent.index[-1], recent.values[-1]),
            xytext=(-60, 20), textcoords='offset points',
            color='#00d4ff', fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#00d4ff'))

ax.annotate(f'Day 30: ${future_mean.values[-1]:.2f}',
            xy=(future_dates[-1], future_mean.values[-1]),
            xytext=(-80, -30), textcoords='offset points',
            color='#f59e0b', fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#f59e0b'))

ax.set_xlabel('Date'); ax.set_ylabel('Price (USD)')
ax.legend(facecolor='#1e2d45', labelcolor='#e2e8f0', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'arima_05_future_forecast.png'), dpi=150)
plt.show()
print("  Saved: arima_05_future_forecast.png")

# ── Plot 5: Compare ARIMA vs ML models (summary)
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Model Comparison — ARIMA vs ML Models', fontsize=13, fontweight='bold', color='#e2e8f0')

models = ['Linear Reg\n(R²=0.999)', 'Random Forest\n(57.4% acc)', f'ARIMA{best_order}\n(Rolling)']
maes   = [1.82, None, mae_roll]
colors = ['#00d4ff', '#10b981', '#f59e0b']

# Only regression-comparable (MAE)
reg_models = ['Linear Reg', 'Poly Reg', f'ARIMA{best_order}']
reg_maes   = [1.82, 2.14, mae_roll]
reg_colors = ['#00d4ff', '#7c3aed', '#f59e0b']

bars = ax.bar(reg_models, reg_maes, color=reg_colors, width=0.4, edgecolor='none')
for bar, v in zip(bars, reg_maes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'${v:.2f}', ha='center', va='bottom', color='#e2e8f0', fontsize=11, fontweight='bold')

ax.set_title('MAE Comparison (USD) — Lower is Better', color='#e2e8f0', fontsize=11)
ax.set_ylabel('MAE (USD)', color='#64748b')
ax.tick_params(colors='#64748b')
for spine in ax.spines.values(): spine.set_color('#1e2d45')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'arima_06_model_comparison.png'), dpi=150)
plt.show()
print("  Saved: arima_06_model_comparison.png")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("   ARIMA MODEL — FINAL SUMMARY")
print("=" * 60)
print(f"\n  Best ARIMA Order : {best_order}")
print(f"  p (AR terms)     : {best_order[0]}  -- uses last {best_order[0]} prices")
print(f"  d (differencing) : {best_order[1]}  -- applied {best_order[1]}x to make stationary")
print(f"  q (MA terms)     : {best_order[2]}  -- uses last {best_order[2]} forecast errors")
print(f"\n  Walk-Forward Forecast Metrics:")
print(f"  MAE  = {mae_roll:.4f}  (avg error in $)")
print(f"  RMSE = {rmse_roll:.4f}")
print(f"  MAPE = {mape_roll:.2f}%  (avg % error)")
print(f"\n  30-Day Future Forecast:")
print(f"  Starting price   : ${recent.values[-1]:.2f}")
print(f"  Forecast Day 30  : ${future_mean.values[-1]:.2f}")
print(f"  95% CI range     : ${future_ci.iloc[-1,0]:.2f} -- ${future_ci.iloc[-1,1]:.2f}")
print(f"\n  Plots saved to: {PLOTS}")
print(f"  arima_01_acf_pacf.png")
print(f"  arima_02_full_forecast.png")
print(f"  arima_03_test_zoom.png")
print(f"  arima_04_residuals.png")
print(f"  arima_05_future_forecast.png")
print(f"  arima_06_model_comparison.png")
print("\n" + "=" * 60)
print("  ARIMA COMPLETE!")
print("=" * 60)