import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

BASE = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction'
PLOTS = os.path.join(BASE, 'outputs', 'plots')
os.makedirs(PLOTS, exist_ok=True)

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

# ── PLOT 1: Decision Tree Visualization ───────────────────────────────
dt = DecisionTreeClassifier(max_depth=3, random_state=42)  # depth=3 for clean visual
dt.fit(X_train_s, y_train_clf)

plt.figure(figsize=(20, 8))
plot_tree(dt,
          feature_names=feature_cols,
          class_names=['DOWN', 'UP'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Structure (max_depth=3)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '11_decision_tree_plot.png'), dpi=150)
plt.show()
print('Decision Tree plot saved!')

# ── PLOT 2: Decision Tree - Actual vs Predicted ───────────────────────
dt5 = DecisionTreeClassifier(max_depth=5, random_state=42)
dt5.fit(X_train_s, y_train_clf)
pred_dt = dt5.predict(X_test_s)

plt.figure(figsize=(12, 4))
plt.plot(y_test_clf.values[:100], color='steelblue', label='Actual (0=DOWN, 1=UP)', linewidth=1.5)
plt.plot(pred_dt[:100],           color='red',       label='Predicted', linewidth=1, linestyle='--', alpha=0.7)
plt.title('Decision Tree - Actual vs Predicted (first 100 test samples)', fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Class (0=DOWN / 1=UP)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '12_decision_tree_actual_vs_predicted.png'), dpi=150)
plt.show()
print('Decision Tree Actual vs Predicted plot saved!')

# ── PLOT 3: Neural Network - Training Loss Curve ──────────────────────
mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                        max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
mlp_clf.fit(X_train_s, y_train_clf)

plt.figure(figsize=(10, 5))
plt.plot(mlp_clf.loss_curve_,         color='steelblue', label='Training Loss',   linewidth=2)
plt.plot(mlp_clf.validation_scores_,  color='orange',    label='Validation Score', linewidth=2)
plt.title('Neural Network - Training Loss & Validation Score', fontsize=13, fontweight='bold')
plt.xlabel('Epochs (Iterations)')
plt.ylabel('Loss / Score')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '13_neural_network_loss_curve.png'), dpi=150)
plt.show()
print('Neural Network loss curve saved!')

# ── PLOT 4: Neural Network - Actual vs Predicted ─────────────────────
pred_mlp_clf = mlp_clf.predict(X_test_s)

plt.figure(figsize=(12, 4))
plt.plot(y_test_clf.values[:100], color='steelblue', label='Actual (0=DOWN, 1=UP)', linewidth=1.5)
plt.plot(pred_mlp_clf[:100],      color='green',     label='Predicted', linewidth=1, linestyle='--', alpha=0.7)
plt.title('Neural Network Classifier - Actual vs Predicted (first 100 test samples)', fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Class (0=DOWN / 1=UP)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '14_neural_network_actual_vs_predicted.png'), dpi=150)
plt.show()
print('Neural Network Actual vs Predicted plot saved!')

# ── PLOT 5: Neural Network Regressor - Price Prediction ──────────────
mlp_reg = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                       max_iter=500, random_state=42, early_stopping=True)
mlp_reg.fit(X_train_s, y_train_reg)
pred_mlp_reg = mlp_reg.predict(X_test_s)

plt.figure(figsize=(14, 5))
plt.plot(y_test_reg.values[:200], color='steelblue', label='Actual Price',          linewidth=1.5)
plt.plot(pred_mlp_reg[:200],      color='green',     label='Neural Net Predicted',   linewidth=1, linestyle='--')
plt.title('Neural Network Regressor - Actual vs Predicted Price', fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '15_neural_network_price_prediction.png'), dpi=150)
plt.show()
print('Neural Network price prediction plot saved!')

# ── PLOT 6: All Models Confusion Matrix (including Neural Network) ────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (name, pred) in zip(axes, [('Decision Tree',    pred_dt),
                                    ('Neural Network',   pred_mlp_clf)]):
    cm = confusion_matrix(y_test_clf, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['DOWN','UP'], yticklabels=['DOWN','UP'])
    ax.set_title(f'{name} Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '16_dt_nn_confusion_matrices.png'), dpi=150)
plt.show()
print('Confusion matrices saved!')

print('\nAll extra plots saved to:', PLOTS)