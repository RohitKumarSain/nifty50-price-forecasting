import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit Page Config
st.set_page_config(page_title="Nifty 50 Predictor", layout="centered")

st.title("📈 Nifty 50 Next-Day Predictor")
st.markdown("Real-time Linear Regression model trained on **20 Years of Raw OHLCV Data**.")

# Data Fetcher & Feature Processor
@st.cache_data(ttl=3600)
def fetch_nifty_data():
    """Fetch 20 years of Nifty 50 data from current date."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=20 * 365)
    
    df = yf.download("^NSEI", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
    
    # Flatten multi-level columns if returned by modern yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def prepare_baseline_data(df, target_col):
    """Prepare OHLCV features and target variable (shifted by -1)."""
    data = df.copy()
    data["Target"] = data[target_col].shift(-1)
    
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean_df = data[feature_cols + ["Target"]].dropna()
    
    X = clean_df[feature_cols]
    y = clean_df["Target"]
    
    return X, y

# UI Selection
target_selection = st.radio(
    "Select Target Price to Predict:",
    options=["Open", "Close"],
    horizontal=True
)

if st.button("🚀 Fetch Data & Predict", type="primary"):
    with st.spinner("Downloading 20 years of Nifty 50 data & training Linear Regression model..."):
        raw_df = fetch_nifty_data()
        
        if raw_df.empty:
            st.error("Failed to fetch market data. Please check your connection.")
        else:
            # 1. Prepare Features & Target
            X, y = prepare_baseline_data(raw_df, target_selection)
            
            # 2. Chronological Train/Test Split (80/20)
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            # 3. Train Pipeline (StandardScaler + LinearRegression)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ])
            pipeline.fit(X_train, y_train)
            
            # 4. Evaluate Test Metrics
            y_pred = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # 5. Predict Next-Day Value using the Latest Available Row
            latest_row = raw_df[["Open", "High", "Low", "Close", "Volume"]].tail(1)
            prediction = pipeline.predict(latest_row)[0]
            
            last_recorded_val = raw_df[target_selection].iloc[-1]
            last_date = raw_df.index[-1].strftime("%Y-%m-%d")
            delta = prediction - last_recorded_val
            
            st.success("🎉 Prediction & Evaluation Complete!")
            
            # 6. Display Metrics
            st.subheader(f"🎯 Target: Next-Day {target_selection} Price")
            col1, col2 = st.columns(2)
            col1.metric(
                label=f"Last Recorded {target_selection} ({last_date})", 
                value=f"₹{last_recorded_val:,.2f}"
            )
            col2.metric(
                label=f"Predicted Next-Day {target_selection}", 
                value=f"₹{prediction:,.2f}",
                delta=f"₹{delta:,.2f} ({(delta / last_recorded_val) * 100:.2f}%)"
            )
            
            st.divider()
            
            st.subheader("📊 Model Performance Metrics (Test Set)")
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("RMSE", f"₹{rmse:,.2f}")
            m3.metric("MAE", f"₹{mae:,.2f}")
            
            # 7. Optional plot for actual vs prediction overview
            st.divider()
            st.subheader(f"📈 Test Period Performance ({target_selection})")
            chart_data = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}, index=y_test.index)
            st.line_chart(chart_data)