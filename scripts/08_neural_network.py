import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
import os

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'

df = pd.read_csv(os.path.join(BASE, 'data', 'AAPL_features.csv'), parse_dates=['Date'])

feature_cols = ['Open','High','Low','Close','Volume',
                'Lag_1','Lag_2','Lag_3','MA_5','MA_10','MA_20','Return']
X     = df[feature_cols]
y_clf = df['Target_Clf']
y_reg = df['Target_Reg']

split_idx = int(len(df) * 0.80)
X_train, X_test         = X.iloc[:split_idx],     X.iloc[split_idx:]
y_train_clf, y_test_clf = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# MLP Classifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                        max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
mlp_clf.fit(X_train_s, y_train_clf)
pred_clf = mlp_clf.predict(X_test_s)
print('=== Neural Network Classifier ===')
print(f'Accuracy: {accuracy_score(y_test_clf, pred_clf):.4f}')
print(f'F1 Score: {f1_score(y_test_clf, pred_clf):.4f}')

# MLP Regressor
mlp_reg = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                       max_iter=500, random_state=42, early_stopping=True)
mlp_reg.fit(X_train_s, y_train_reg)
pred_reg = mlp_reg.predict(X_test_s)
print('\n=== Neural Network Regressor ===')
print(f'MAE : {mean_absolute_error(y_test_reg, pred_reg):.4f}')
print(f'R²  : {r2_score(y_test_reg, pred_reg):.4f}')