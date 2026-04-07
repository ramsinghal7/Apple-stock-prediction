import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'
PLOTS = os.path.join(BASE, 'outputs', 'plots')
os.makedirs(PLOTS, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, 'data', 'AAPL_features.csv'), parse_dates=['Date'])

# Plot 1: Price Over Time
plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Close'], color='steelblue', linewidth=0.8, label='Close Price')
plt.title('Apple (AAPL) Closing Price Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date'); plt.ylabel('Price (USD)'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '01_price_over_time.png'), dpi=150); plt.show()

# Plot 2: Moving Averages
plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Close'],  color='grey',   alpha=0.4, linewidth=0.6, label='Close')
plt.plot(df['Date'], df['MA_5'],   color='blue',   linewidth=1.2, label='MA 5')
plt.plot(df['Date'], df['MA_10'],  color='orange', linewidth=1.2, label='MA 10')
plt.plot(df['Date'], df['MA_20'],  color='red',    linewidth=1.2, label='MA 20')
plt.title('Moving Averages (MA5, MA10, MA20)', fontsize=14, fontweight='bold')
plt.xlabel('Date'); plt.ylabel('Price (USD)'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '02_moving_averages.png'), dpi=150); plt.show()

# Plot 3: Distribution of Close Price
plt.figure(figsize=(8, 4))
sns.histplot(df['Close'], bins=60, kde=True, color='steelblue')
plt.title('Distribution of AAPL Close Price', fontsize=13, fontweight='bold')
plt.xlabel('Close Price (USD)'); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '03_close_distribution.png'), dpi=150); plt.show()

# Plot 4: Distribution of Daily Returns
plt.figure(figsize=(8, 4))
sns.histplot(df['Return'].dropna(), bins=80, kde=True, color='darkorange')
plt.title('Distribution of Daily Returns', fontsize=13, fontweight='bold')
plt.xlabel('Daily Return (%)'); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '04_returns_distribution.png'), dpi=150); plt.show()

# Plot 5: Correlation Heatmap
num_cols = ['Open','High','Low','Close','Volume','Lag_1','Lag_2',
            'MA_5','MA_10','MA_20','Return','Target_Reg']
corr = df[num_cols].corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, annot_kws={'size': 8})
plt.title('Correlation Heatmap of All Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '05_correlation_heatmap.png'), dpi=150); plt.show()

# Plot 6: Volume Over Time
plt.figure(figsize=(14, 4))
plt.bar(df['Date'], df['Volume'], color='teal', alpha=0.5, width=1)
plt.title('AAPL Trading Volume Over Time', fontsize=13, fontweight='bold')
plt.xlabel('Date'); plt.ylabel('Volume'); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '06_volume.png'), dpi=150); plt.show()

print('All EDA plots saved to:', PLOTS)