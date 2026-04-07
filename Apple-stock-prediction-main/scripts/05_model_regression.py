import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'
PLOTS = os.path.join(BASE, 'outputs', 'plots')
os.makedirs(PLOTS, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, 'data', 'AAPL_features.csv'), parse_dates=['Date'])

feature_cols = ['Open','High','Low','Close','Volume',
                'Lag_1','Lag_2','Lag_3','MA_5','MA_10','MA_20','Return']
X = df[feature_cols]
y = df['Target_Reg']

split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
print('=== Linear Regression ===')
print(f'MAE : {mean_absolute_error(y_test, y_pred_lr):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}')
print(f'R²  : {r2_score(y_test, y_pred_lr):.4f}')

# Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_s)
X_test_poly  = poly.transform(X_test_s)
lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)
print('\n=== Polynomial Regression (degree=2) ===')
print(f'MAE : {mean_absolute_error(y_test, y_pred_poly):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_poly)):.4f}')
print(f'R²  : {r2_score(y_test, y_pred_poly):.4f}')

# Plot
plt.figure(figsize=(14, 5))
plt.plot(y_test.values[:200],  color='steelblue', label='Actual Price')
plt.plot(y_pred_lr[:200],      color='red',       linestyle='--', label='Linear Reg')
plt.plot(y_pred_poly[:200],    color='green',     linestyle=':',  label='Poly Reg')
plt.title('Actual vs Predicted — Regression Models', fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '07_regression_predictions.png'), dpi=150)
plt.show()