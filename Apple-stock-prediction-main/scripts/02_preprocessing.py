import pandas as pd
import numpy as np
import os

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'

df = pd.read_csv(os.path.join(BASE, 'data', 'AAPL.csv'))

# Step 1: Convert Date
df['Date'] = pd.to_datetime(df['Date'])

# Step 2: Sort chronologically
df = df.sort_values('Date').reset_index(drop=True)

# Step 3: Handle Missing Values
print('Missing values before:', df.isnull().sum().sum())
df.ffill(inplace=True)
df.bfill(inplace=True)
print('Missing values after:', df.isnull().sum().sum())

# Step 4: Remove Duplicates
df.drop_duplicates(subset='Date', keep='first', inplace=True)
print('Shape after dedup:', df.shape)

print('\nPreprocessing complete!')
print(df.info())

df.to_csv(os.path.join(BASE, 'data', 'AAPL_clean.csv'), index=False)
print('Saved: AAPL_clean.csv')