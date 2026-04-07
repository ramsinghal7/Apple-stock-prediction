import pandas as pd
import numpy as np
import os

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'

df = pd.read_csv(os.path.join(BASE, 'data', 'AAPL.csv'))

print('Shape:', df.shape)
print('\nFirst 5 rows:')
print(df.head())
print('\nColumn data types:')
print(df.dtypes)
print('\nMissing values per column:')
print(df.isnull().sum())
print('\nDuplicate rows:', df.duplicated().sum())
print('\nStatistical summary:')
print(df.describe())
print('\nDate range:', df['Date'].min(), 'to', df['Date'].max())