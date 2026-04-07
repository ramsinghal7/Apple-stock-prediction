import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'

df = pd.read_csv(os.path.join(BASE, 'data', 'AAPL_features.csv'), parse_dates=['Date'])

feature_cols = ['Open','High','Low','Close','Volume',
                'Lag_1','Lag_2','Lag_3','MA_5','MA_10','MA_20','Return']

X     = df[feature_cols]
y_reg = df['Target_Reg']
y_clf = df['Target_Clf']

split_idx = int(len(df) * 0.80)

X_train, X_test         = X.iloc[:split_idx],     X.iloc[split_idx:]
y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
y_train_clf, y_test_clf = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]

print(f'Total samples    : {len(df)}')
print(f'Training samples : {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)')
print(f'Testing samples  : {len(X_test)}  ({len(X_test)/len(df)*100:.1f}%)')
print(f'Train period : {df["Date"].iloc[0].date()} -> {df["Date"].iloc[split_idx-1].date()}')
print(f'Test period  : {df["Date"].iloc[split_idx].date()} -> {df["Date"].iloc[-1].date()}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print('\nFeature scaling done using StandardScaler')
print('X_train shape:', X_train_scaled.shape)
print('X_test shape :', X_test_scaled.shape)