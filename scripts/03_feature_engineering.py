import pandas as pd
import numpy as np
import os

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'

df = pd.read_csv(os.path.join(BASE, 'data', 'AAPL_clean.csv'), parse_dates=['Date'])

# Lag Features
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)

# Moving Averages
df['MA_5']  = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()

# Daily Return
df['Return'] = df['Close'].pct_change()

# Regression Target
df['Target_Reg'] = df['Close'].shift(-1)

# Classification Target
df['Target_Clf'] = (df['Target_Reg'] > df['Close']).astype(int)

# Drop NaNs
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print('Shape after feature engineering:', df.shape)
print('\nNew columns:', [c for c in df.columns if c not in
    ['Date','Open','High','Low','Close','Adj Close','Volume']])
print('\nSample (last 5 rows):')
print(df.tail())

df.to_csv(os.path.join(BASE, 'data', 'AAPL_features.csv'), index=False)
print('Saved: AAPL_features.csv')