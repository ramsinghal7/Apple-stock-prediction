import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, f1_score,
                             precision_score, recall_score)
import os

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'
PLOTS = os.path.join(BASE, 'outputs', 'plots')
os.makedirs(PLOTS, exist_ok=True)

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

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Regression models
lr = LinearRegression().fit(X_train_s, y_train_reg)
y_lr = lr.predict(X_test_s)

poly = PolynomialFeatures(degree=2, include_bias=False)
lr_poly = LinearRegression().fit(poly.fit_transform(X_train_s), y_train_reg)
y_poly = lr_poly.predict(poly.transform(X_test_s))

mlp_reg = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                       random_state=42, early_stopping=True)
mlp_reg.fit(X_train_s, y_train_reg)
y_mlp = mlp_reg.predict(X_test_s)

reg_results = {
    'Linear Regression':    {'MAE': mean_absolute_error(y_test_reg, y_lr),   'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_lr)),   'R2': r2_score(y_test_reg, y_lr)},
    'Polynomial Regression':{'MAE': mean_absolute_error(y_test_reg, y_poly), 'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_poly)), 'R2': r2_score(y_test_reg, y_poly)},
    'Neural Net Regressor': {'MAE': mean_absolute_error(y_test_reg, y_mlp),  'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_mlp)),  'R2': r2_score(y_test_reg, y_mlp)},
}
print('\n=== REGRESSION COMPARISON ===')
print(pd.DataFrame(reg_results).T.round(4).to_string())

# Classification models
clf_results = {}
for name, model in [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Decision Tree',       DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('Random Forest',       RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)),
    ('Neural Net Clf',      MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                          random_state=42, early_stopping=True)),
]:
    model.fit(X_train_s, y_train_clf)
    p = model.predict(X_test_s)
    clf_results[name] = {
        'Accuracy':  accuracy_score(y_test_clf, p),
        'Precision': precision_score(y_test_clf, p),
        'Recall':    recall_score(y_test_clf, p),
        'F1':        f1_score(y_test_clf, p)
    }

clf_df = pd.DataFrame(clf_results).T.round(4)
print('\n=== CLASSIFICATION COMPARISON ===')
print(clf_df.to_string())

clf_df[['Accuracy','F1']].plot(kind='bar', figsize=(10, 5), colormap='Set2')
plt.title('Classification Model Comparison', fontweight='bold')
plt.ylabel('Score'); plt.xticks(rotation=20); plt.ylim(0.4, 1.0)
plt.legend(loc='lower right'); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '09_model_comparison.png'), dpi=150)
plt.show()