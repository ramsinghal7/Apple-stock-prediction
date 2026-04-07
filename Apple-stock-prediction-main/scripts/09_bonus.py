import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import os

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'
PLOTS = os.path.join(BASE, 'outputs', 'plots')
os.makedirs(PLOTS, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, 'data', 'AAPL_features.csv'), parse_dates=['Date'])

feature_cols = ['Open','High','Low','Close','Volume',
                'Lag_1','Lag_2','Lag_3','MA_5','MA_10','MA_20','Return']
X = df[feature_cols]
y = df['Target_Clf']

split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_s, y_train)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
importances.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — Random Forest', fontweight='bold')
plt.xlabel('Importance Score'); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '10_feature_importance.png'), dpi=150)
plt.show()

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=tscv, scoring='accuracy')
print('CV Scores:', cv_scores.round(4))
print(f'Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Hyperparameter Tuning
param_grid = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'max_features':      ['sqrt', 'log2'],
}
rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), param_grid,
    n_iter=20, cv=TimeSeriesSplit(n_splits=3),
    scoring='accuracy', random_state=42, n_jobs=-1)
rs.fit(X_train_s, y_train)

print('\nBest Parameters:', rs.best_params_)
print(f'Best CV Accuracy: {rs.best_score_:.4f}')

best_pred = rs.best_estimator_.predict(X_test_s)
print(f'Tuned RF — Test Accuracy: {accuracy_score(y_test, best_pred):.4f}')
print(f'Tuned RF — F1 Score     : {f1_score(y_test, best_pred):.4f}')