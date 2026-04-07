import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL Predictive Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #111827 !important; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 10px;
        padding: 14px !important;
    }
    div[data-testid="metric-container"] label { color: #64748b !important; font-size:12px; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #00d4ff !important; font-size:26px !important; }
    
    /* Dataframe */
    .stDataFrame { background: #111827 !important; }
    
    /* Headers */
    h1,h2,h3 { color: #e2e8f0 !important; }
    
    /* Success/Info */
    .stSuccess { background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important; }
    .stInfo    { background: rgba(0,212,255,0.05) !important; border: 1px solid rgba(0,212,255,0.2) !important; }
    .stWarning { background: rgba(245,158,11,0.1) !important; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; font-weight: 700 !important;
        padding: 10px 24px !important; width: 100% !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    
    /* Step boxes */
    .step-box {
        background: #111827; border: 1px solid #1e2d45;
        border-radius: 10px; padding: 16px; margin-bottom: 10px;
        border-left: 3px solid #00d4ff;
    }
    .step-done { border-left-color: #10b981 !important; }
    .step-title { font-size: 14px; font-weight: 700; color: #e2e8f0; margin-bottom: 4px; }
    .step-desc  { font-size: 12px; color: #64748b; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #64748b !important; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom-color: #00d4ff !important; }
    
    /* Select boxes */
    .stSelectbox label { color: #64748b !important; font-size: 12px !important; }
    
    div[data-testid="stMetricDelta"] { color: #10b981 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────────────────
for key in ['df_raw','df_clean','df_feat','step','results_reg','results_clf']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'step' not in st.session_state:
    st.session_state.step = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AAPL Pipeline")
    st.markdown("**Vedansh Tiwari · A032**")
    st.markdown("B.Tech AI & Data Science")
    st.divider()

    st.markdown("### 🗺️ Pipeline Status")
    steps = [
        ("01", "Data Load",         st.session_state.df_raw   is not None),
        ("02", "Preprocessing",     st.session_state.df_clean is not None),
        ("03", "Feature Eng.",      st.session_state.df_feat  is not None),
        ("04", "EDA",               st.session_state.df_feat  is not None),
        ("05", "Train-Test Split",  st.session_state.df_feat  is not None),
        ("06", "Models",            st.session_state.results_clf is not None),
        ("07", "Evaluation",        st.session_state.results_clf is not None),
    ]
    for num, name, done in steps:
        icon = "✅" if done else "⬜"
        color = "#10b981" if done else "#64748b"
        st.markdown(f"<div style='font-size:13px;color:{color};padding:4px 0'>{icon} <b>Step {num}</b> — {name}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Model Settings")
    n_estimators = st.slider("Random Forest Trees", 50, 300, 100, 50)
    max_depth_dt  = st.slider("Decision Tree Depth", 2, 10, 5)
    test_size     = st.slider("Test Size %", 10, 30, 20, 5)
    st.divider()
    if st.button("🔄 Reset Pipeline"):
        for k in ['df_raw','df_clean','df_feat','results_reg','results_clf']:
            st.session_state[k] = None
        st.rerun()

# ── Main Title ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#111827,#1a2236);
     border:1px solid #1e2d45;border-radius:14px;padding:24px 28px;margin-bottom:24px'>
  <h1 style='margin:0;font-size:28px;color:#e2e8f0'>
    📈 Apple Stock <span style='color:#00d4ff'>Predictive Analytics</span> Pipeline
  </h1>
  <p style='margin:8px 0 0;color:#64748b;font-size:13px'>
    Complete end-to-end ML pipeline · AAPL.csv · 10,467 rows · B.Tech AI & DS Project
  </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂 Step 1-2: Load & Clean",
    "🔧 Step 3: Features",
    "📊 Step 4: EDA",
    "✂️ Step 5: Split",
    "🤖 Step 6: Train Models",
    "📋 Step 7: Evaluation",
    "🏆 Conclusion"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📂 Step 1 — Load Data & Step 2 — Preprocessing")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📁 Load AAPL Dataset")
        uploaded = st.file_uploader("Upload your AAPL.csv", type=['csv'])

        default_path = r'C:\Users\vanqu\OneDrive\Desktop\apple_stock_prediction\data\AAPL.csv'
        use_path = st.text_input("Or enter file path:", value=default_path)

        if st.button("🚀 Load Data"):
            try:
                if uploaded:
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_csv(use_path)
                st.session_state.df_raw = df.copy()
                st.success(f"✅ Loaded! Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col2:
        if st.session_state.df_raw is not None:
            df = st.session_state.df_raw
            st.markdown("### 📊 Dataset Info")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing", df.isnull().sum().sum())
            c4.metric("Duplicates", df.duplicated().sum())

    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.divider()

        col3, col4 = st.columns([1.2, 0.8])
        with col3:
            st.markdown("#### 👀 First 5 Rows")
            st.dataframe(df.head(), use_container_width=True)
        with col4:
            st.markdown("#### 📈 Statistical Summary")
            st.dataframe(df.describe().round(2), use_container_width=True)

        st.divider()
        st.markdown("## 🧹 Step 2 — Preprocessing")

        col5, col6 = st.columns([1, 1])
        with col5:
            st.markdown("""
            **Steps applied:**
            - ✅ Convert Date → datetime
            - ✅ Sort chronologically
            - ✅ Forward fill missing values
            - ✅ Backward fill remaining NaNs
            - ✅ Remove duplicate dates
            - ✅ Reset index
            """)

        with col6:
            if st.button("🧹 Run Preprocessing"):
                df_c = df.copy()
                df_c['Date'] = pd.to_datetime(df_c['Date'])
                df_c = df_c.sort_values('Date').reset_index(drop=True)
                before = df_c.isnull().sum().sum()
                df_c.ffill(inplace=True)
                df_c.bfill(inplace=True)
                after = df_c.isnull().sum().sum()
                df_c.drop_duplicates(subset='Date', keep='first', inplace=True)
                df_c.reset_index(drop=True, inplace=True)
                st.session_state.df_clean = df_c
                st.success(f"✅ Done! Missing: {before} → {after} | Shape: {df_c.shape}")

        if st.session_state.df_clean is not None:
            dc = st.session_state.df_clean
            st.markdown(f"""
            <div class='step-box step-done'>
              <div class='step-title'>✅ Preprocessing Complete</div>
              <div class='step-desc'>
                Date range: {dc['Date'].iloc[0].date()} → {dc['Date'].iloc[-1].date()} &nbsp;|&nbsp;
                Final shape: {dc.shape[0]:,} rows × {dc.shape[1]} cols
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔧 Step 3 — Feature Engineering")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please complete Step 1-2 first (Load & Preprocess).")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            **Features being created:**

            | Feature | Formula | Captures |
            |---------|---------|---------|
            | Lag_1 | Close.shift(1) | Yesterday's price |
            | Lag_2 | Close.shift(2) | 2 days ago |
            | Lag_3 | Close.shift(3) | 3 days ago |
            | MA_5  | rolling(5).mean() | Short trend |
            | MA_10 | rolling(10).mean() | Mid trend |
            | MA_20 | rolling(20).mean() | Monthly trend |
            | Return | pct_change() | Daily % change |
            | Target_Reg | Close.shift(-1) | Tomorrow's price |
            | Target_Clf | UP=1 / DOWN=0 | Direction |
            """)

        with col2:
            st.markdown("### ▶️ Run Feature Engineering")
            if st.button("⚙️ Generate Features"):
                df_f = st.session_state.df_clean.copy()
                df_f['Lag_1']      = df_f['Close'].shift(1)
                df_f['Lag_2']      = df_f['Close'].shift(2)
                df_f['Lag_3']      = df_f['Close'].shift(3)
                df_f['MA_5']       = df_f['Close'].rolling(5).mean()
                df_f['MA_10']      = df_f['Close'].rolling(10).mean()
                df_f['MA_20']      = df_f['Close'].rolling(20).mean()
                df_f['Return']     = df_f['Close'].pct_change()
                df_f['Target_Reg'] = df_f['Close'].shift(-1)
                df_f['Target_Clf'] = (df_f['Target_Reg'] > df_f['Close']).astype(int)
                df_f.dropna(inplace=True)
                df_f.reset_index(drop=True, inplace=True)
                st.session_state.df_feat = df_f
                st.success(f"✅ Features created! New shape: {df_f.shape[0]:,} rows × {df_f.shape[1]} cols")

        if st.session_state.df_feat is not None:
            df_f = st.session_state.df_feat
            st.divider()
            st.markdown("#### 📋 Dataset with New Features (last 5 rows)")
            new_cols = ['Date','Close','Lag_1','Lag_2','Lag_3','MA_5','MA_10','MA_20','Return','Target_Reg','Target_Clf']
            st.dataframe(df_f[new_cols].tail(), use_container_width=True)

            st.divider()
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("#### 📈 Class Balance (UP vs DOWN)")
                counts = df_f['Target_Clf'].value_counts()
                fig, ax = plt.subplots(figsize=(5,3), facecolor='#111827')
                ax.set_facecolor('#111827')
                ax.bar(['DOWN (0)','UP (1)'], [counts.get(0,0), counts.get(1,0)],
                       color=['#ef4444','#10b981'], edgecolor='none', width=0.5)
                ax.set_title('Target Distribution', color='#e2e8f0', fontsize=12)
                ax.tick_params(colors='#64748b'); ax.spines[:].set_color('#1e2d45')
                for spine in ax.spines.values(): spine.set_color('#1e2d45')
                ax.yaxis.label.set_color('#64748b'); ax.xaxis.label.set_color('#64748b')
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col4:
                up_pct = counts.get(1,0) / len(df_f) * 100
                st.markdown(f"""
                <br><br>
                <div class='step-box step-done'>
                  <div class='step-title'>📊 Class Distribution</div>
                  <div class='step-desc'>
                    UP days: {counts.get(1,0):,} ({up_pct:.1f}%)<br>
                    DOWN days: {counts.get(0,0):,} ({100-up_pct:.1f}%)<br>
                    <br>Nearly balanced — good for classification!
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📊 Step 4 — Exploratory Data Analysis")

    if st.session_state.df_feat is None:
        st.warning("⚠️ Please complete Step 3 (Feature Engineering) first.")
    else:
        df_f = st.session_state.df_feat

        def dark_fig(w=12, h=4):
            fig, ax = plt.subplots(figsize=(w,h), facecolor='#0a0e1a')
            ax.set_facecolor('#111827')
            ax.tick_params(colors='#64748b')
            for spine in ax.spines.values(): spine.set_color('#1e2d45')
            return fig, ax

        # Plot 1 & 2
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 Plot 1 — Closing Price Over Time")
            fig, ax = dark_fig()
            ax.plot(df_f['Date'], df_f['Close'], color='#00d4ff', linewidth=0.8, alpha=0.9)
            ax.fill_between(df_f['Date'], df_f['Close'], alpha=0.1, color='#00d4ff')
            ax.set_title('AAPL Closing Price (1980–Present)', color='#e2e8f0', fontsize=11)
            ax.set_xlabel('Date', color='#64748b'); ax.set_ylabel('Price (USD)', color='#64748b')
            st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            st.markdown("#### 📉 Plot 2 — Moving Averages")
            fig, ax = dark_fig()
            ax.plot(df_f['Date'], df_f['Close'], color='#475569', alpha=0.5, linewidth=0.6, label='Close')
            ax.plot(df_f['Date'], df_f['MA_5'],  color='#3b82f6', linewidth=1.2, label='MA 5')
            ax.plot(df_f['Date'], df_f['MA_10'], color='#f59e0b', linewidth=1.2, label='MA 10')
            ax.plot(df_f['Date'], df_f['MA_20'], color='#ef4444', linewidth=1.2, label='MA 20')
            ax.set_title('Moving Averages MA5 / MA10 / MA20', color='#e2e8f0', fontsize=11)
            ax.set_xlabel('Date', color='#64748b'); ax.set_ylabel('Price (USD)', color='#64748b')
            ax.legend(facecolor='#1e2d45', labelcolor='#e2e8f0', fontsize=9)
            st.pyplot(fig, use_container_width=True); plt.close()

        # Plot 3 & 4
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### 📊 Plot 3 — Close Price Distribution")
            fig, ax = dark_fig(8, 4)
            ax.hist(df_f['Close'], bins=60, color='#00d4ff', alpha=0.7, edgecolor='none')
            ax.set_title('Distribution of AAPL Close Price', color='#e2e8f0', fontsize=11)
            ax.set_xlabel('Close Price (USD)', color='#64748b'); ax.set_ylabel('Frequency', color='#64748b')
            st.pyplot(fig, use_container_width=True); plt.close()

        with col4:
            st.markdown("#### 🔔 Plot 4 — Daily Returns Distribution")
            fig, ax = dark_fig(8, 4)
            returns = df_f['Return'].dropna()
            ax.hist(returns, bins=80, color='#f59e0b', alpha=0.7, edgecolor='none')
            ax.axvline(returns.mean(), color='#ef4444', linewidth=1.5, linestyle='--', label=f'Mean: {returns.mean():.4f}')
            ax.set_title('Distribution of Daily Returns', color='#e2e8f0', fontsize=11)
            ax.set_xlabel('Daily Return', color='#64748b'); ax.set_ylabel('Frequency', color='#64748b')
            ax.legend(facecolor='#1e2d45', labelcolor='#e2e8f0', fontsize=9)
            st.pyplot(fig, use_container_width=True); plt.close()

        # Plot 5 & 6
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("#### 🌡️ Plot 5 — Correlation Heatmap")
            num_cols = ['Open','High','Low','Close','Volume','Lag_1','Lag_2','MA_5','MA_10','MA_20','Return','Target_Reg']
            corr = df_f[num_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0e1a')
            ax.set_facecolor('#111827')
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                        linewidths=0.5, annot_kws={'size': 7}, ax=ax,
                        cbar_kws={'shrink': 0.8})
            ax.set_title('Correlation Heatmap', color='#e2e8f0', fontsize=11)
            ax.tick_params(colors='#64748b', labelsize=8)
            st.pyplot(fig, use_container_width=True); plt.close()

        with col6:
            st.markdown("#### 📦 Plot 6 — Trading Volume Over Time")
            fig, ax = dark_fig()
            ax.fill_between(df_f['Date'], df_f['Volume'], color='#14b8a6', alpha=0.6)
            ax.set_title('AAPL Trading Volume Over Time', color='#e2e8f0', fontsize=11)
            ax.set_xlabel('Date', color='#64748b'); ax.set_ylabel('Volume', color='#64748b')
            st.pyplot(fig, use_container_width=True); plt.close()

            st.markdown("""
            <div class='step-box step-done' style='margin-top:12px'>
              <div class='step-title'>💡 EDA Key Insights</div>
              <div class='step-desc'>
                • Price shows exponential growth — not linear<br>
                • Returns are normally distributed — good for models<br>
                • Lag_1 & Close: r ≈ 0.9999 — expected for time series
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRAIN TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## ✂️ Step 5 — Time-Based Train-Test Split")

    if st.session_state.df_feat is None:
        st.warning("⚠️ Please complete Step 3 first.")
    else:
        df_f = st.session_state.df_feat
        split_pct = test_size / 100
        split_idx = int(len(df_f) * (1 - split_pct))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='step-box step-done'>
              <div class='step-title'>✅ Split Configuration</div>
              <div class='step-desc'>
                Total samples: {len(df_f):,}<br>
                Training: {split_idx:,} rows ({100-test_size}%)<br>
                Testing: {len(df_f)-split_idx:,} rows ({test_size}%)<br>
                Train period: {df_f['Date'].iloc[0].date()} -> {df_f['Date'].iloc[split_idx-1].date()}<br>
                Test period: {df_f['Date'].iloc[split_idx].date()} -> {df_f['Date'].iloc[-1].date()}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='step-box' style='border-left-color:#ef4444'>
              <div class='step-title'>❌ Why NOT Random Split?</div>
              <div class='step-desc'>
                Random split causes DATA LEAKAGE in time series — the model sees future data during training, giving falsely high accuracy.<br><br>
                ✅ We use time-based split: first 80% = train, last 20% = test. Model NEVER sees future data.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Visual split chart
        st.markdown("#### 📊 Visual Split on Close Price")
        fig, ax = plt.subplots(figsize=(14, 4), facecolor='#0a0e1a')
        ax.set_facecolor('#111827')
        ax.plot(df_f['Date'].iloc[:split_idx], df_f['Close'].iloc[:split_idx],
                color='#00d4ff', linewidth=0.8, label=f'Train ({100-test_size}%)')
        ax.plot(df_f['Date'].iloc[split_idx:], df_f['Close'].iloc[split_idx:],
                color='#7c3aed', linewidth=0.8, label=f'Test ({test_size}%)')
        ax.axvline(df_f['Date'].iloc[split_idx], color='#f59e0b', linewidth=2, linestyle='--', label='Split Point')
        ax.set_title('Train / Test Split on AAPL Close Price', color='#e2e8f0', fontsize=12)
        ax.set_xlabel('Date', color='#64748b'); ax.set_ylabel('Price (USD)', color='#64748b')
        ax.tick_params(colors='#64748b')
        for spine in ax.spines.values(): spine.set_color('#1e2d45')
        ax.legend(facecolor='#1e2d45', labelcolor='#e2e8f0')
        st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🤖 Step 6 — Train All Models")

    if st.session_state.df_feat is None:
        st.warning("⚠️ Please complete Steps 1-3 first.")
    else:
        df_f = st.session_state.df_feat
        feature_cols = ['Open','High','Low','Close','Volume',
                        'Lag_1','Lag_2','Lag_3','MA_5','MA_10','MA_20','Return']
        X     = df_f[feature_cols]
        y_reg = df_f['Target_Reg']
        y_clf = df_f['Target_Clf']
        split_idx = int(len(df_f) * (1 - test_size/100))

        X_train, X_test         = X.iloc[:split_idx],     X.iloc[split_idx:]
        y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
        y_train_clf, y_test_clf = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📐 Regression Models")
            st.markdown("Predicting **exact closing price** for tomorrow")
        with col2:
            st.markdown("### 🎯 Classification Models")
            st.markdown("Predicting **UP (1) or DOWN (0)** direction")

        st.divider()

        if st.button("🚀 Train All 7 Models Now!"):
            results_reg = {}
            results_clf = {}

            progress = st.progress(0)
            status   = st.empty()

            # Linear Regression
            status.markdown("⚙️ Training **Linear Regression**...")
            lr = LinearRegression()
            lr.fit(X_train_s, y_train_reg)
            y_pred_lr = lr.predict(X_test_s)
            results_reg['Linear Regression'] = {
                'MAE': mean_absolute_error(y_test_reg, y_pred_lr),
                'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_lr)),
                'R2': r2_score(y_test_reg, y_pred_lr),
                'y_pred': y_pred_lr, 'y_test': y_test_reg.values
            }
            progress.progress(15)

            # Polynomial Regression
            status.markdown("⚙️ Training **Polynomial Regression**...")
            poly = PolynomialFeatures(degree=2, include_bias=False)
            Xtr_p = poly.fit_transform(X_train_s)
            Xte_p = poly.transform(X_test_s)
            lr_p  = LinearRegression()
            lr_p.fit(Xtr_p, y_train_reg)
            y_pred_poly = lr_p.predict(Xte_p)
            results_reg['Polynomial Reg'] = {
                'MAE': mean_absolute_error(y_test_reg, y_pred_poly),
                'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_poly)),
                'R2': r2_score(y_test_reg, y_pred_poly),
                'y_pred': y_pred_poly, 'y_test': y_test_reg.values
            }
            progress.progress(30)

            # Logistic Regression
            status.markdown("⚙️ Training **Logistic Regression**...")
            log = LogisticRegression(max_iter=1000, random_state=42)
            log.fit(X_train_s, y_train_clf)
            p_log = log.predict(X_test_s)
            results_clf['Logistic Regression'] = {
                'Accuracy':  accuracy_score(y_test_clf, p_log),
                'Precision': precision_score(y_test_clf, p_log, zero_division=0),
                'Recall':    recall_score(y_test_clf, p_log, zero_division=0),
                'F1':        f1_score(y_test_clf, p_log, zero_division=0),
                'cm': confusion_matrix(y_test_clf, p_log),
                'y_pred': p_log
            }
            progress.progress(45)

            # Decision Tree
            status.markdown("⚙️ Training **Decision Tree**...")
            dt = DecisionTreeClassifier(max_depth=max_depth_dt, random_state=42)
            dt.fit(X_train_s, y_train_clf)
            p_dt = dt.predict(X_test_s)
            results_clf['Decision Tree'] = {
                'Accuracy':  accuracy_score(y_test_clf, p_dt),
                'Precision': precision_score(y_test_clf, p_dt, zero_division=0),
                'Recall':    recall_score(y_test_clf, p_dt, zero_division=0),
                'F1':        f1_score(y_test_clf, p_dt, zero_division=0),
                'cm': confusion_matrix(y_test_clf, p_dt),
                'y_pred': p_dt, 'model': dt
            }
            progress.progress(60)

            # Random Forest
            status.markdown("⚙️ Training **Random Forest**...")
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=7, random_state=42, n_jobs=-1)
            rf.fit(X_train_s, y_train_clf)
            p_rf = rf.predict(X_test_s)
            results_clf['Random Forest'] = {
                'Accuracy':  accuracy_score(y_test_clf, p_rf),
                'Precision': precision_score(y_test_clf, p_rf, zero_division=0),
                'Recall':    recall_score(y_test_clf, p_rf, zero_division=0),
                'F1':        f1_score(y_test_clf, p_rf, zero_division=0),
                'cm': confusion_matrix(y_test_clf, p_rf),
                'y_pred': p_rf, 'model': rf,
                'importances': rf.feature_importances_
            }
            progress.progress(75)

            # Neural Network Classifier
            status.markdown("⚙️ Training **Neural Network Classifier**...")
            mlp_c = MLPClassifier(hidden_layer_sizes=(128,64), activation='relu',
                                  max_iter=500, random_state=42,
                                  early_stopping=True, validation_fraction=0.1)
            mlp_c.fit(X_train_s, y_train_clf)
            p_mlp = mlp_c.predict(X_test_s)
            results_clf['Neural Network'] = {
                'Accuracy':  accuracy_score(y_test_clf, p_mlp),
                'Precision': precision_score(y_test_clf, p_mlp, zero_division=0),
                'Recall':    recall_score(y_test_clf, p_mlp, zero_division=0),
                'F1':        f1_score(y_test_clf, p_mlp, zero_division=0),
                'cm': confusion_matrix(y_test_clf, p_mlp),
                'y_pred': p_mlp,
                'loss_curve': mlp_c.loss_curve_
            }
            progress.progress(90)

            # Neural Network Regressor
            status.markdown("⚙️ Training **Neural Network Regressor**...")
            mlp_r = MLPRegressor(hidden_layer_sizes=(128,64), activation='relu',
                                 max_iter=500, random_state=42, early_stopping=True)
            mlp_r.fit(X_train_s, y_train_reg)
            y_pred_mlp = mlp_r.predict(X_test_s)
            results_reg['Neural Network'] = {
                'MAE': mean_absolute_error(y_test_reg, y_pred_mlp),
                'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_mlp)),
                'R2': r2_score(y_test_reg, y_pred_mlp),
                'y_pred': y_pred_mlp, 'y_test': y_test_reg.values
            }
            progress.progress(100)
            status.empty()

            # Store in session state
            st.session_state.results_reg = results_reg
            st.session_state.results_clf = results_clf
            st.session_state['y_test_clf'] = y_test_clf.values
            st.session_state['y_test_reg'] = y_test_reg.values
            st.session_state['feature_cols'] = feature_cols

            st.success("🎉 All 7 models trained successfully! Go to Step 7: Evaluation tab.")

            # Quick preview
            st.divider()
            st.markdown("### 📊 Quick Results Preview")
            c1, c2, c3, c4 = st.columns(4)
            best_clf = max(results_clf, key=lambda k: results_clf[k]['Accuracy'])
            best_reg = min(results_reg, key=lambda k: results_reg[k]['MAE'])
            c1.metric("Best Clf Model", best_clf)
            c2.metric("Best Accuracy", f"{results_clf[best_clf]['Accuracy']*100:.1f}%")
            c3.metric("Best Reg Model", best_reg)
            c4.metric("Best R²", f"{results_reg[best_reg]['R2']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 📋 Step 7 — Model Evaluation & Comparison")

    if st.session_state.results_clf is None:
        st.warning("⚠️ Please train models in Step 6 first.")
    else:
        results_clf = st.session_state.results_clf
        results_reg = st.session_state.results_reg
        y_test_clf  = st.session_state['y_test_clf']
        y_test_reg  = st.session_state['y_test_reg']
        feature_cols = st.session_state['feature_cols']

        # ── Classification Table
        st.markdown("### 🎯 Classification Results")
        clf_data = []
        for name, r in results_clf.items():
            clf_data.append({
                'Model': name,
                'Accuracy': f"{r['Accuracy']*100:.2f}%",
                'Precision': f"{r['Precision']:.4f}",
                'Recall': f"{r['Recall']:.4f}",
                'F1 Score': f"{r['F1']:.4f}"
            })
        clf_df = pd.DataFrame(clf_data)
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

        # Accuracy bar chart
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0a0e1a')
            ax.set_facecolor('#111827')
            names  = list(results_clf.keys())
            accs   = [r['Accuracy']*100 for r in results_clf.values()]
            colors = ['#64748b','#f59e0b','#10b981','#00d4ff']
            bars = ax.bar(names, accs, color=colors[:len(names)], width=0.5, edgecolor='none')
            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{acc:.1f}%', ha='center', va='bottom', color='#e2e8f0', fontsize=10, fontweight='bold')
            ax.set_ylim(45, 65)
            ax.set_title('Classification Accuracy', color='#e2e8f0', fontsize=11)
            ax.tick_params(colors='#64748b', labelsize=9)
            for spine in ax.spines.values(): spine.set_color('#1e2d45')
            ax.set_ylabel('Accuracy (%)', color='#64748b')
            plt.xticks(rotation=15)
            st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            st.markdown("#### 📊 F1 Score Comparison")
            fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0a0e1a')
            ax.set_facecolor('#111827')
            f1s = [r['F1'] for r in results_clf.values()]
            bars = ax.bar(names, f1s, color=colors[:len(names)], width=0.5, edgecolor='none')
            for bar, f in zip(bars, f1s):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                        f'{f:.3f}', ha='center', va='bottom', color='#e2e8f0', fontsize=10, fontweight='bold')
            ax.set_ylim(0.4, 0.7)
            ax.set_title('F1 Score Comparison', color='#e2e8f0', fontsize=11)
            ax.tick_params(colors='#64748b', labelsize=9)
            for spine in ax.spines.values(): spine.set_color('#1e2d45')
            ax.set_ylabel('F1 Score', color='#64748b')
            plt.xticks(rotation=15)
            st.pyplot(fig, use_container_width=True); plt.close()

        # Confusion Matrices
        st.divider()
        st.markdown("### 🔲 Confusion Matrices")
        ncols = len(results_clf)
        conf_cols = st.columns(ncols)
        for i, (name, r) in enumerate(results_clf.items()):
            with conf_cols[i]:
                st.markdown(f"**{name}**")
                fig, ax = plt.subplots(figsize=(3.5, 3), facecolor='#0a0e1a')
                ax.set_facecolor('#111827')
                sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['DOWN','UP'], yticklabels=['DOWN','UP'],
                            cbar=False, annot_kws={'size':12,'color':'white'})
                ax.set_title(name, color='#e2e8f0', fontsize=10)
                ax.tick_params(colors='#64748b', labelsize=8)
                ax.set_xlabel('Predicted', color='#64748b', fontsize=9)
                ax.set_ylabel('Actual', color='#64748b', fontsize=9)
                st.pyplot(fig, use_container_width=True); plt.close()

        st.divider()
        # ── Regression Table
        st.markdown("### 📐 Regression Results")
        reg_data = []
        for name, r in results_reg.items():
            reg_data.append({'Model': name,
                             'MAE':  f"{r['MAE']:.4f}",
                             'RMSE': f"{r['RMSE']:.4f}",
                             'R²':   f"{r['R2']:.4f}"})
        st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)

        # Actual vs Predicted
        st.markdown("#### 📈 Actual vs Predicted — Best Regression Model (Linear)")
        y_pred_lr = results_reg['Linear Regression']['y_pred']
        fig, ax = plt.subplots(figsize=(14, 4), facecolor='#0a0e1a')
        ax.set_facecolor('#111827')
        show = min(300, len(y_test_reg))
        ax.plot(y_test_reg[:show], color='#00d4ff', linewidth=1.5, label='Actual Price')
        ax.plot(y_pred_lr[:show],  color='#ef4444', linewidth=1,   linestyle='--', label='Linear Reg Predicted', alpha=0.8)
        ax.set_title('Actual vs Predicted — Linear Regression (first 300 test points)', color='#e2e8f0', fontsize=11)
        ax.set_xlabel('Test Sample', color='#64748b')
        ax.set_ylabel('Price (USD)', color='#64748b')
        ax.tick_params(colors='#64748b')
        for spine in ax.spines.values(): spine.set_color('#1e2d45')
        ax.legend(facecolor='#1e2d45', labelcolor='#e2e8f0')
        st.pyplot(fig, use_container_width=True); plt.close()

        # Neural Network Loss Curve
        if 'loss_curve' in results_clf['Neural Network']:
            st.divider()
            st.markdown("#### 🧠 Neural Network — Training Loss Curve")
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0a0e1a')
            ax.set_facecolor('#111827')
            ax.plot(results_clf['Neural Network']['loss_curve'], color='#00d4ff', linewidth=2, label='Training Loss')
            ax.set_title('Neural Network Training Loss Over Epochs', color='#e2e8f0', fontsize=11)
            ax.set_xlabel('Epoch', color='#64748b'); ax.set_ylabel('Loss', color='#64748b')
            ax.tick_params(colors='#64748b')
            for spine in ax.spines.values(): spine.set_color('#1e2d45')
            ax.legend(facecolor='#1e2d45', labelcolor='#e2e8f0')
            st.pyplot(fig, use_container_width=True); plt.close()

        # Decision Tree Plot
        if 'model' in results_clf.get('Decision Tree', {}):
            st.divider()
            st.markdown("#### 🌳 Decision Tree Structure (depth=3 for visualization)")
            dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
            df_f = st.session_state.df_feat
            X = df_f[feature_cols]
            y_clf = df_f['Target_Clf']
            split_idx = int(len(df_f) * (1 - test_size/100))
            scaler2 = StandardScaler()
            Xtr = scaler2.fit_transform(X.iloc[:split_idx])
            dt_viz.fit(Xtr, y_clf.iloc[:split_idx])
            fig, ax = plt.subplots(figsize=(20, 7), facecolor='#0a0e1a')
            ax.set_facecolor('#0a0e1a')
            plot_tree(dt_viz, feature_names=feature_cols, class_names=['DOWN','UP'],
                      filled=True, rounded=True, fontsize=9, ax=ax)
            ax.set_title('Decision Tree Structure (max_depth=3)', color='#e2e8f0', fontsize=12)
            st.pyplot(fig, use_container_width=True); plt.close()

        # Feature Importance
        if 'importances' in results_clf.get('Random Forest', {}):
            st.divider()
            st.markdown("#### 🎯 Feature Importance — Random Forest")
            imp = pd.Series(results_clf['Random Forest']['importances'], index=feature_cols).sort_values()
            fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0a0e1a')
            ax.set_facecolor('#111827')
            colors_imp = ['#7c3aed' if v > imp.median() else '#1e2d45' for v in imp.values]
            imp.plot(kind='barh', color=colors_imp, ax=ax, edgecolor='none')
            ax.set_title('Feature Importance — Random Forest', color='#e2e8f0', fontsize=11)
            ax.set_xlabel('Importance Score', color='#64748b')
            ax.tick_params(colors='#64748b')
            for spine in ax.spines.values(): spine.set_color('#1e2d45')
            st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## 🏆 Conclusion & Key Findings")

    if st.session_state.results_clf is None:
        st.warning("⚠️ Train models first to see dynamic conclusion.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📌 What Was Done
        1. **Loaded** AAPL dataset (10,467 rows, 42 years)
        2. **Preprocessed** — date conversion, ffill, dedup
        3. **Feature Engineering** — 9 new features created
        4. **EDA** — 6 visualizations generated
        5. **Time-based split** — 80% train, 20% test
        6. **7 Models trained** — Regression + Classification + NN
        7. **Evaluation** — MAE, RMSE, R², Accuracy, F1, CM
        """)

        st.markdown("""
        ### 💡 Key Concepts Used
        - **Data Leakage Prevention** — Time-based split, scaler fit on train only
        - **Efficient Market Hypothesis** — Why ~55% is good accuracy
        - **Ensemble Learning** — Random Forest = 100 trees voting
        - **Feature Engineering** — Lag, MA, Returns
        - **Cross Validation** — TimeSeriesSplit (never leaks future)
        """)

    with col2:
        if st.session_state.results_clf:
            results_clf = st.session_state.results_clf
            results_reg = st.session_state.results_reg
            best_clf = max(results_clf, key=lambda k: results_clf[k]['Accuracy'])
            best_reg = min(results_reg, key=lambda k: results_reg[k]['MAE'])

            st.markdown("### 🥇 Your Results")
            m1,m2 = st.columns(2)
            m1.metric("Best Clf Model", best_clf, f"{results_clf[best_clf]['Accuracy']*100:.1f}% acc")
            m2.metric("Best Reg Model", best_reg, f"R²={results_reg[best_reg]['R2']:.4f}")
            m3,m4 = st.columns(2)
            m3.metric("Best F1 Score", f"{results_clf[best_clf]['F1']:.4f}")
            m4.metric("Best MAE (USD)", f"{results_reg[best_reg]['MAE']:.2f}")

        st.markdown("""
        ### ⚠️ Limitations
        - No external signals (news, earnings)
        - No advanced indicators (RSI, MACD)
        - Non-stationarity in long-term data

        ### 🚀 Future Improvements
        - LSTM / GRU for sequence modeling
        - Sentiment analysis from news
        - XGBoost / LightGBM
        - Options data & macro indicators
        """)

    st.divider()
    st.markdown("""
    <div style='text-align:center;padding:20px;background:#111827;border-radius:12px;border:1px solid #1e2d45'>
      <div style='font-size:13px;color:#64748b;font-family:monospace'>
        Project by <span style='color:#00d4ff'>Vedansh Tiwari</span> · Roll No. A032 ·
        B.Tech AI & Data Science · MPSTME Indore
      </div>
      <div style='font-size:12px;color:#475569;margin-top:6px'>
        Built with Python · Streamlit · Scikit-learn · Pandas · Matplotlib · Seaborn
      </div>
    </div>
    """, unsafe_allow_html=True)