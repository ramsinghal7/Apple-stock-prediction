import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
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

results = {}

def eval_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    results[name] = {
        'Accuracy':  accuracy_score(y_te, pred),
        'Precision': precision_score(y_te, pred),
        'Recall':    recall_score(y_te, pred),
        'F1':        f1_score(y_te, pred)
    }
    print(f'\n=== {name} ===')
    for k, v in results[name].items():
        print(f'{k:10s}: {v:.4f}')
    return model, pred

log_model, pred_log = eval_model('Logistic Regression',
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_s, X_test_s, y_train, y_test)

dt_model, pred_dt = eval_model('Decision Tree',
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X_train_s, X_test_s, y_train, y_test)

rf_model, pred_rf = eval_model('Random Forest',
    RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1),
    X_train_s, X_test_s, y_train, y_test)

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, pred) in zip(axes, [('Logistic Reg', pred_log),
                                    ('Decision Tree', pred_dt),
                                    ('Random Forest', pred_rf)]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '08_confusion_matrices.png'), dpi=150)
plt.show()