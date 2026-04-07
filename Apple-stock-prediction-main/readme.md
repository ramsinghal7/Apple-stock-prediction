# 📈 Apple Stock Predictive Analytics Pipeline

Complete end-to-end ML + ARIMA pipeline for AAPL stock price prediction.

## 👥 Team
| Name | Roll No |
|------|---------|
| Vedansh Tiwari | A032 |
| Ram Singhal | A023 |

## 🎓 Course
B.Tech AI & Data Science | Predictive Analytics | MPSTME Indore

## 📊 Dataset
AAPL.csv — 10,467 rows × 7 columns — 1980 to 2022

## 🗂️ Project structure new
apple_stock_prediction/
├── data/               # Dataset (not uploaded — add your own AAPL.csv)
├── scripts/
│   ├── 01_data_understanding.py
│   ├── 02_preprocessing.py
│   ├── 03_feature_engineering.py
│   ├── 04_eda.py
│   ├── 05_train_test_split.py
│   ├── 05_model_regression.py
│   ├── 06_model_classification.py
│   ├── 07_neural_network.py
│   ├── 08_evaluation.py
│   ├── 09_bonus.py
│   └── 10_arima.py
├── outputs/
│   └── plots/          # All generated charts
├── notebooks/
│   └── analysis.ipynb
└── requirements.txt
## 🤖 Models Built
- Linear Regression (R² = 0.9991)
- Polynomial Regression
- Logistic Regression
- Decision Tree
- Random Forest (Best: 57.4% accuracy)
- Neural Network (MLP)
- ARIMA(2,1,4) — Time Series

## ▶️ How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run in order
python scripts/01_data_understanding.py
python scripts/02_preprocessing.py
python scripts/03_feature_engineering.py
python scripts/04_eda.py
python scripts/05_train_test_split.py
python scripts/05_model_regression.py
python scripts/06_model_classification.py
python scripts/07_neural_network.py
python scripts/08_evaluation.py
python scripts/09_bonus.py
python scripts/10_arima.py
```

## 📦 Requirements  
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
joblib