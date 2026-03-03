import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import requests
from io import StringIO
import pytz

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Volatility Visualizer",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Crypto Volatility Visualizer")
st.markdown("### Simulating Market Swings with Mathematics for AI and Python")
st.markdown("*Using sine, cosine, random noise, and integrals to model cryptocurrency volatility*")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# GITHUB CSV URL
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/Nihith007/Crypto-Volatility-Visualizer"
    "/refs/heads/main/btcusd_1-min_data.csv"
)

TIMEZONE_OPTIONS = {
    "🇮🇳 India (IST) UTC+5:30":        "Asia/Kolkata",
    "🌍 UTC (Universal)":               "UTC",
    "🇺🇸 New York (EST/EDT)":           "America/New_York",
    "🇺🇸 Los Angeles (PST/PDT)":        "America/Los_Angeles",
    "🇬🇧 London (GMT/BST)":             "Europe/London",
    "🇪🇺 Paris / Berlin (CET/CEST)":    "Europe/Paris",
    "🇦🇪 Dubai (GST) UTC+4":            "Asia/Dubai",
    "🇸🇬 Singapore (SGT) UTC+8":        "Asia/Singapore",
    "🇯🇵 Tokyo (JST) UTC+9":            "Asia/Tokyo",
    "🇦🇺 Sydney (AEDT) UTC+11":         "Australia/Sydney",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — MODE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Dashboard Mode")
mode = st.sidebar.radio(
    "Choose your analysis mode:",
    ["📊 Real Bitcoin Data", "🧮 Mathematical Simulation", "🔍 Compare Both"],
)
st.sidebar.markdown("---")

st.sidebar.header("🌐 Timezone")
selected_tz_label = st.sidebar.selectbox(
    "Display times in:",
    list(TIMEZONE_OPTIONS.keys()),
    index=0,
)
selected_tz = TIMEZONE_OPTIONS[selected_tz_label]
tz = pytz.timezone(selected_tz)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS & DATA PREPARATION (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────
def apply_timezone(df, tz):
    df = df.copy()
    if df["Timestamp"].dt.tz is None:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC").dt.tz_convert(tz)
    else:
        df["Timestamp"] = df["Timestamp"].dt.tz_convert(tz)
    return df

@st.cache_data(show_spinner="📡 Fetching data...")
def fetch_github_data():
    try:
        r = requests.get(GITHUB_CSV_URL, timeout=60)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text)), None
    except Exception as e:
        return None, str(e)

def normalize_dataframe(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for col in ["Timestamp", "timestamp", "Date", "date", "Time", "time"]:
        if col in df.columns:
            try:
                df["Timestamp"] = pd.to_datetime(df[col], unit="s")
            except Exception:
                df["Timestamp"] = pd.to_datetime(df[col])
            break
    else:
        df["Timestamp"] = pd.date_range("2012-01-01", periods=len(df), freq="min")

    for src in ["Close", "close", "Price", "price"]:
        if src in df.columns:
            df["Price"] = pd.to_numeric(df[src], errors="coerce")
            break

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Price" in df.columns:
        for c in ["Open", "Close"]:
            if c not in df.columns: df[c] = df["Price"]
        if "High" not in df.columns: df["High"] = df["Price"] * 1.002
        if "Low" not in df.columns: df["Low"]  = df["Price"] * 0.998
    if "Volume" not in df.columns:
        df["Volume"] = np.random.uniform(0, 50, len(df))

    df = df.dropna(subset=["Timestamp", "Price"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df

def load_real_data(source="github", uploaded_file=None):
    if source == "upload" and uploaded_file is not None:
        return normalize_dataframe(pd.read_csv(uploaded_file)), None
    raw, err = fetch_github_data()
    if err or raw is None:
        # Fallback sample data (30 days)
        np.random.seed(42)
        minutes = 30 * 24 * 60
        timestamps = pd.date_range(end=datetime.now(), periods=minutes, freq="min")
        prices = np.cumsum(np.random.normal(0, 50, minutes)) + 45000
        return pd.DataFrame({
            "Timestamp": timestamps, "Price": prices,
            "High": prices + 100, "Low": prices - 100, "Volume": np.random.uniform(0, 50, minutes)
        }), err
    return normalize_dataframe(raw), None

def generate_mathematical_data(pattern, amp, freq, drift_val, noise, days, tz):
    hours = days * 24
    t = np.linspace(0, days, hours)
    base, sigma = 45000, max(noise, 1)
    if pattern == "Sine Wave (Smooth Cycles)":
        prices = base + amp * np.sin(2 * np.pi * freq * t / days)
    elif "Noise" in pattern:
        prices = base + amp * np.sin(2 * np.pi * freq * t / days) + np.cumsum(np.random.normal(0, sigma, hours))
    else:
        prices = base + np.cumsum(np.random.normal(0, sigma, hours))
        
    prices = prices + drift_val * t / days
    now_tz = datetime.now(pytz.timezone(tz))
    stamps = [now_tz - timedelta(hours=hours-i) for i in range(hours)]
    return pd.DataFrame({
        "Timestamp": stamps, "Price": np.maximum(prices, 100),
        "High": prices + np.random.uniform(50, 200, hours), 
        "Low": prices - np.random.uniform(50, 200, hours), 
        "Volume": np.random.uniform(1000, 5000, hours)
    })

# ─────────────────────────────────────────────────────────────────────────────
# UPDATED DATE RANGE FILTER (Up to 30 Days)
# ─────────────────────────────────────────────────────────────────────────────
def date_range_filter(df, key_prefix="main"):
    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()
    total_days = (max_date - min_date).days
    
    if total_days < 1: return df

    st.sidebar.subheader("📅 Date Range Filter")
    
    # Updated max_value to 30 as requested
    days_back = st.sidebar.slider(
        "Show last N days",
        min_value=1,
        max_value=min(max(total_days, 1), 30),
        value=min(30, total_days),
        key=f"{key_prefix}_slider"
    )
    
    start_date = max_date - timedelta(days=days_back)
    st.sidebar.info(f"📆 Showing: {start_date} to {max_date}")
    return df[df["Timestamp"].dt.date >= start_date]

# ─────────────────────────────────────────────────────────────────────────────
# METRICS & VISUALIZATIONS (Stage 5 & 6)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_volatility(df):
    returns = df["Price"].pct_change().dropna()
    return returns.std() * np.sqrt(len(returns)) * 100

def show_sidebar_metrics(df):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Key Metrics")
    st.sidebar.metric("Volatility Index", f"{calculate_volatility(df):.2f}%")
    st.sidebar.metric("Current Price", f"${df['Price'].iloc[-1]:,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if mode == "📊 Real Bitcoin Data":
    st.header("📊 Real Bitcoin Data Analysis")
    
    data_source = st.sidebar.radio("Source:", ["🌐 GitHub (Auto-fetch)", "📂 Upload CSV"])
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) if data_source == "📂 Upload CSV" else None
    
    df_raw, err = load_real_data("github" if data_source == "🌐 GitHub (Auto-fetch)" else "upload", uploaded_file)
    df_tz = apply_timezone(df_raw, selected_tz)
    df = date_range_filter(df_tz, "real")
    show_sidebar_metrics(df)

    st.subheader("📈 Price Over Time")
    st.plotly_chart(px.line(df, x="Timestamp", y="Price"), use_container_width=True)

    st.subheader("📉 High vs Low Comparison")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Timestamp"], y=df["High"], name="High", line=dict(color="green")))
    fig2.add_trace(go.Scatter(x=df["Timestamp"], y=df["Low"], name="Low", line=dict(color="red"), fill='tonexty'))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📦 Volume Analysis")
    st.plotly_chart(px.bar(df, x="Timestamp", y="Volume"), use_container_width=True)

elif mode == "🧮 Mathematical Simulation":
    st.header("🧮 Mathematical Simulation")
    # Increased slider range to 30 days for consistency
    num_days = st.sidebar.slider("Number of Days", 1, 30, 30)
    df_m = generate_mathematical_data("Sine + Noise", 1000, 1.0, 10, 100, num_days, selected_tz)
    show_sidebar_metrics(df_m)
    st.plotly_chart(px.line(df_m, x="Timestamp", y="Price", title="Simulated Market Swings"), use_container_width=True)

else:
    st.header("🔍 Comparison: Real vs Simulated")
    df_raw, _ = load_real_data()
    df_tz = apply_timezone(df_raw, selected_tz)
    df_r = date_range_filter(df_tz, "compare")
    df_m = generate_mathematical_data("Sine + Noise", 1000, 1.0, 10, 100, 30, selected_tz)
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.line(df_r, x="Timestamp", y="Price", title="Real Data"), use_container_width=True)
    with c2: st.plotly_chart(px.line(df_m, x="Timestamp", y="Price", title="Simulated Data"), use_container_width=True)

st.markdown("---")
st.markdown("**FinTechLab Pvt. Ltd.** | FA-2: Dashboarding and Deployment")
