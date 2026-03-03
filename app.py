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
        prices = base + amp * np.sin(2 * np.pi * freq * t / days)
        
    prices = prices + drift_val * t / days
    now_tz = datetime.now(pytz.timezone(tz))
    stamps = [now_tz - timedelta(hours=hours-i) for i in range(hours)]
    return pd.DataFrame({
        "Timestamp": stamps, "Price": np.maximum(prices, 100),
        "High": prices + np.random.uniform(50, 200, hours), 
        "Low": prices - np.random.uniform(50, 200, hours), 
        "Volume": np.random.uniform(1000, 5000, hours)
    })

def calculate_volatility(df):
    returns = df["Price"].pct_change().dropna()
    return returns.std() * np.sqrt(len(returns)) * 100

def calculate_drift_metric(df):
    return (df["Price"].iloc[-1] - df["Price"].iloc[0]) / df["Price"].iloc[0] * 100

# ─────────────────────────────────────────────────────────────────────────────
# MODE: REAL BITCOIN DATA (Updated with 30-Day Slider & Statistics)
# ─────────────────────────────────────────────────────────────────────────────
if mode == "📊 Real Bitcoin Data":
    st.header("📊 Real Bitcoin Data Analysis")
    
    data_source = st.sidebar.radio("Source:", ["🌐 GitHub (Auto-fetch)", "📂 Upload CSV"])
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) if data_source == "📂 Upload CSV" else None
    
    show_volatility_bands = st.sidebar.checkbox("Show Volatility Bands", value=True)
    show_volume = st.sidebar.checkbox("Show Volume Analysis", value=True)

    df_raw, err = load_real_data("github" if data_source == "🌐 GitHub (Auto-fetch)" else "upload", uploaded_file)
    df_tz = apply_timezone(df_raw, selected_tz)
    
    # 30-DAY SLIDER LOGIC
    max_avail_days = (df_tz["Timestamp"].max().date() - df_tz["Timestamp"].min().date()).days
    st.sidebar.subheader("📅 Date Range Filter")
    days_back = st.sidebar.slider("Show last N days", 1, min(max(max_avail_days, 1), 30), min(30, max_avail_days))
    df = df_tz[df_tz["Timestamp"] >= (df_tz["Timestamp"].max() - timedelta(days=days_back))]
    
    # Sidebar Metrics
    st.sidebar.markdown("---")
    st.sidebar.metric("Volatility Index", f"{calculate_volatility(df):.2f}%")
    st.sidebar.metric("Current Price", f"${df['Price'].iloc[-1]:,.2f}")

    # Layout: Price Chart and Statistics
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Bitcoin Price Over Time")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df["Timestamp"], y=df["Price"], name="Close Price", line=dict(color="#1f77b4")))
        if show_volatility_bands:
            rm = df["Price"].rolling(20).mean(); rs = df["Price"].rolling(20).std()
            fig1.add_trace(go.Scatter(x=df["Timestamp"], y=rm+2*rs, mode='lines', line=dict(width=0), showlegend=False))
            fig1.add_trace(go.Scatter(x=df["Timestamp"], y=rm-2*rs, mode='lines', fill='tonexty', fillcolor='rgba(31,119,180,0.1)', name="Volatility Band"))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("📊 Price Statistics")
        stats = pd.DataFrame({
            "Metric": ["Min", "Max", "Avg", "Data Points"],
            "Value": [f"${df['Price'].min():,.2f}", f"${df['Price'].max():,.2f}", f"${df['Price'].mean():,.2f}", f"{len(df):,}"]
        })
        st.table(stats)
        st.markdown("##### Price Distribution")
        st.plotly_chart(px.histogram(df, x="Price", nbins=30), use_container_width=True)

    # High vs Low Comparison
    st.subheader("📉 High vs Low Price Comparison")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Timestamp"], y=df["High"], name="High", line=dict(color="green", width=1)))
    fig2.add_trace(go.Scatter(x=df["Timestamp"], y=df["Low"], name="Low", line=dict(color="red", width=1), fill='tonexty', fillcolor='rgba(0,255,0,0.05)'))
    st.plotly_chart(fig2, use_container_width=True)

    # Stable vs Volatile Periods
    st.subheader("🔀 Stable vs Volatile Analysis")
    col3, col4 = st.columns(2)
    df["Rolling_Vol"] = df["Price"].rolling(window=20).std()
    threshold = df["Rolling_Vol"].median()
    df["Status"] = df["Rolling_Vol"].apply(lambda x: "Volatile" if x > threshold else "Stable")
    
    with col3:
        st.plotly_chart(px.line(df, x="Timestamp", y="Rolling_Vol", title="Rolling Volatility (Std Dev)", color_discrete_sequence=['orange']), use_container_width=True)
    with col4:
        st.plotly_chart(px.pie(df, names="Status", title="Period Distribution", color="Status", color_discrete_map={"Stable":"lightgreen", "Volatile":"salmon"}), use_container_width=True)

    if show_volume:
        st.subheader("📦 Volume Analysis")
        st.plotly_chart(px.bar(df, x="Timestamp", y="Volume", color_discrete_sequence=['steelblue']), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODE: MATHEMATICAL SIMULATION (Original 7-day default preserved)
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "🧮 Mathematical Simulation":
    st.header("🧮 Mathematical Simulation")
    pattern = st.sidebar.selectbox("Pattern:", ["Sine Wave (Smooth Cycles)", "Sine + Noise (Realistic Market)", "Random Noise (Chaotic Jumps)"])
    amp = st.sidebar.slider("Amplitude", 100, 5000, 1000)
    freq = st.sidebar.slider("Frequency", 0.1, 5.0, 1.0)
    drift = st.sidebar.slider("Drift", -50, 50, 10)
    noise = st.sidebar.slider("Noise", 0, 500, 100)
    num_days = st.sidebar.slider("Number of Days", 1, 30, 7)
    
    df_m = generate_mathematical_data(pattern, amp, freq, drift, noise, num_days, selected_tz)
    st.plotly_chart(px.line(df_m, x="Timestamp", y="Price", title=f"Simulated {pattern}"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODE: COMPARE BOTH
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.header("🔍 Comparison: Real vs Simulated")
    df_raw, _ = load_real_data()
    df_tz = apply_timezone(df_raw, selected_tz)
    df_r = df_tz.tail(1440) # Last 24 hours for quick compare
    df_m = generate_mathematical_data("Sine + Noise", 1000, 1.0, 10, 100, 1, selected_tz)
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.line(df_r, x="Timestamp", y="Price", title="Real Data (Recent)"), use_container_width=True)
    with c2: st.plotly_chart(px.line(df_m, x="Timestamp", y="Price", title="Simulated Data"), use_container_width=True)

st.markdown("---")
st.markdown("**Crypto Volatility Visualizer** | Mathematics for AI-II | FinTechLab Pvt. Ltd.")
