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
# HELPERS
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

@st.cache_data
def create_sample_bitcoin_data(days=30):
    np.random.seed(42)
    minutes = days * 24 * 60
    timestamps = pd.date_range(end=datetime.now(), periods=minutes, freq="min")
    prices = [45000]
    for _ in range(minutes - 1):
        prices.append(max(100, prices[-1] + np.random.normal(0, 50)))
    prices = np.array(prices)
    return pd.DataFrame({
        "Timestamp": timestamps, "Price": prices,
        "Open": prices * np.random.uniform(0.999, 1.001, minutes),
        "High": prices * np.random.uniform(1.001, 1.005, minutes),
        "Low": prices * np.random.uniform(0.995, 0.999, minutes),
        "Close": prices, "Volume": np.random.uniform(0, 50, minutes),
    })

def load_real_data(source="github", uploaded_file=None):
    if source == "upload" and uploaded_file is not None:
        return normalize_dataframe(pd.read_csv(uploaded_file)), None
    raw, err = fetch_github_data()
    if err or raw is None:
        return create_sample_bitcoin_data(), err
    return normalize_dataframe(raw), None

def generate_mathematical_data(pattern, amp, freq, drift_val, noise, days, tz):
    hours = days * 24
    t = np.linspace(0, days, hours)
    base, sigma = 45000, max(noise, 1)
    if pattern == "Sine Wave (Smooth Cycles)":
        prices = base + amp * np.sin(2 * np.pi * freq * t / days)
    elif pattern == "Cosine Wave (Smooth Cycles)":
        prices = base + amp * np.cos(2 * np.pi * freq * t / days)
    elif pattern == "Random Noise (Chaotic Jumps)":
        prices = base + np.cumsum(np.random.normal(0, sigma, hours))
    elif "Sine + Noise" in pattern:
        prices = (base + amp * np.sin(2 * np.pi * freq * t / days) + np.cumsum(np.random.normal(0, sigma, hours)))
    else:
        prices = base + amp * np.sin(2 * np.pi * freq * t / days) # Default
        
    prices = prices + drift_val * t / days
    now_tz = datetime.now(pytz.timezone(tz))
    stamps = [now_tz - timedelta(hours=hours-i) for i in range(hours)]
    return pd.DataFrame({
        "Timestamp": stamps, "Price": np.maximum(prices, 100),
        "High": prices + 100, "Low": prices - 100, "Volume": np.random.uniform(1000, 5000, hours)
    })

# ─────────────────────────────────────────────────────────────────────────────
# UPDATED 30-DAY DATE FILTER
# ─────────────────────────────────────────────────────────────────────────────
def date_range_filter(df, key_prefix="main"):
    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()
    total_days = (max_date - min_date).days
    if total_days < 1: return df

    st.sidebar.subheader("📅 Date Range Filter")
    # Updated to max_value=30
    days_back = st.sidebar.slider(
        "Show last N days",
        min_value=1,
        max_value=min(max(total_days, 1), 30),
        value=min(30, total_days),
        key=f"{key_prefix}_slider"
    )
    start_date = max_date - timedelta(days=days_back)
    st.sidebar.info(f"📆 {start_date} → {max_date}")
    return df[df["Timestamp"].dt.date >= start_date]

def calculate_volatility(df):
    returns = df["Price"].pct_change().dropna()
    return returns.std() * np.sqrt(len(returns)) * 100

def calculate_drift_metric(df):
    return (df["Price"].iloc[-1] - df["Price"].iloc[0]) / df["Price"].iloc[0] * 100

def show_sidebar_metrics(df):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Key Metrics")
    st.sidebar.metric("Volatility Index", f"{calculate_volatility(df):.2f}%")
    st.sidebar.metric("Average Drift",    f"{calculate_drift_metric(df):+.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────────────────────────────────────
if mode == "📊 Real Bitcoin Data":
    st.header("📊 Real Bitcoin Data Analysis")
    data_source = st.sidebar.radio("Source:", ["🌐 GitHub (Auto-fetch)", "📂 Upload CSV"])
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) if data_source == "📂 Upload CSV" else None
    
    show_volatility_bands = st.sidebar.checkbox("Show Volatility Bands", value=True)
    show_volume = st.sidebar.checkbox("Show Volume Analysis", value=True)

    df_raw, err = load_real_data("github" if data_source == "🌐 GitHub (Auto-fetch)" else "upload", uploaded_file)
    df_tz = apply_timezone(df_raw, selected_tz)
    df = date_range_filter(df_tz, "real")
    show_sidebar_metrics(df)

    # Main Price Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["Timestamp"], y=df["Price"], name="Close Price"))
    st.plotly_chart(fig1, use_container_width=True)

    # High vs Low Comparison (FA-2 Requirement)
    st.subheader("📉 High vs Low Comparison")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Timestamp"], y=df["High"], name="High", line=dict(color="green")))
    fig2.add_trace(go.Scatter(x=df["Timestamp"], y=df["Low"], name="Low", line=dict(color="red"), fill='tonexty'))
    st.plotly_chart(fig2, use_container_width=True)

    if show_volume:
        st.subheader("📦 Volume Analysis")
        st.plotly_chart(px.bar(df, x="Timestamp", y="Volume"), use_container_width=True)

elif mode == "🧮 Mathematical Simulation":
    st.header("🧮 Mathematical Simulation")
    pattern = st.sidebar.selectbox("Pattern:", ["Sine Wave (Smooth Cycles)", "Sine + Noise (Realistic Market)"])
    amp = st.sidebar.slider("Amplitude", 100, 5000, 1000)
    freq = st.sidebar.slider("Frequency", 0.1, 5.0, 1.0)
    drift = st.sidebar.slider("Drift", -50, 50, 10)
    noise = st.sidebar.slider("Noise", 0, 500, 100)
    num_days = st.sidebar.slider("Number of Days", 1, 7, 7) # Kept at 7 as requested
    
    df_m = generate_mathematical_data(pattern, amp, freq, drift, noise, num_days, selected_tz)
    show_sidebar_metrics(df_m)
    st.plotly_chart(px.line(df_m, x="Timestamp", y="Price", title="Simulated Price"), use_container_width=True)

else: # Compare Both
    st.header("🔍 Comparison Mode")
    df_raw, _ = load_real_data()
    df_tz = apply_timezone(df_raw, selected_tz)
    df_r = date_range_filter(df_tz, "compare")
    df_m = generate_mathematical_data("Sine + Noise", 1000, 1.0, 10, 100, 7, selected_tz)
    
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(px.line(df_r, x="Timestamp", y="Price", title="Real Data"), use_container_width=True)
    with col2: st.plotly_chart(px.line(df_m, x="Timestamp", y="Price", title="Simulation"), use_container_width=True)

st.markdown("---")
st.markdown("**FinTechLab Pvt. Ltd.** | Mathematics for AI-II")
