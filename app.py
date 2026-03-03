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
st.markdown("*Using sine, cosine, random noise, and integrals to model cryptocurrency volatility* ")
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
    "🇦🇪 Dubai (GST) UTC+4":            "Asia/Dubai",
    "🇯🇵 Tokyo (JST) UTC+9":            "Asia/Tokyo",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — MODE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Dashboard Mode")
mode = st.sidebar.radio(
    "Choose your analysis mode:",
    ["📊 Real Bitcoin Data", "🧮 Mathematical Simulation", "🔍 Compare Both"],
)

st.sidebar.header("🌐 Timezone")
selected_tz_label = st.sidebar.selectbox(
    "Display times in:",
    list(TIMEZONE_OPTIONS.keys()),
    index=0,
)
selected_tz = TIMEZONE_OPTIONS[selected_tz_label]
tz = pytz.timezone(selected_tz)

# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
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
    for col in ["Timestamp", "timestamp", "Date", "date"]:
        if col in df.columns:
            try:
                df["Timestamp"] = pd.to_datetime(df[col], unit="s")
            except:
                df["Timestamp"] = pd.to_datetime(df[col])
            break
    
    # Rename Close to Price for internal logic 
    if "Close" in df.columns:
        df["Price"] = pd.to_numeric(df["Close"], errors="coerce")
    
    for col in ["High", "Low", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df = df.dropna(subset=["Timestamp", "Price"])
    return df.sort_values("Timestamp").reset_index(drop=True)

def generate_mathematical_data(pattern, amp, freq, drift_val, noise, days, tz_str):
    hours = days * 24
    t = np.linspace(0, days, hours)
    base = 45000
    
    if "Sine" in pattern:
        prices = base + amp * np.sin(2 * np.pi * freq * t / days)
    elif "Cosine" in pattern:
        prices = base + amp * np.cos(2 * np.pi * freq * t / days)
    elif "Random" in pattern:
        prices = base + np.cumsum(np.random.normal(0, noise, hours))
    else:
        prices = base + (amp * np.sin(2 * np.pi * freq * t / days)) + np.cumsum(np.random.normal(0, noise, hours))

    prices = prices + (drift_val * t / days) # Integral/Drift component 
    
    now_tz = datetime.now(pytz.timezone(tz_str))
    stamps = [now_tz - timedelta(hours=hours-i) for i in range(hours)]

    return pd.DataFrame({
        "Timestamp": stamps,
        "Price": np.maximum(prices, 100),
        "High": prices + np.random.uniform(50, 150, hours),
        "Low": prices - np.random.uniform(50, 150, hours),
        "Volume": np.random.uniform(500, 5000, hours)
    })

# ─────────────────────────────────────────────────────────────────────────────
# NEW UPDATED FILTER (30 DAYS)
# ─────────────────────────────────────────────────────────────────────────────
def date_range_filter(df, key_prefix="main"):
    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()
    total_available_days = (max_date - min_date).days

    st.sidebar.subheader("📅 Date Range Filter")
    # Updated to allow up to 30 days if data exists 
    days_back = st.sidebar.slider(
        "Show last N days",
        min_value=1,
        max_value=min(total_available_days, 30) if total_available_days > 0 else 1,
        value=min(total_available_days, 30) if total_available_days > 0 else 1,
        key=f"{key_prefix}_slider"
    )
    start_date = max_date - timedelta(days=days_back)
    return df[df["Timestamp"].dt.date >= start_date]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if mode == "📊 Real Bitcoin Data":
    raw_data, err = fetch_github_data()
    if not err:
        df_clean = normalize_dataframe(raw_data)
        df_tz = apply_timezone(df_clean, tz)
        df = date_range_filter(df_tz, "real")
        
        st.subheader("Bitcoin Price Performance")
        fig = px.line(df, x="Timestamp", y="Price", title="Actual Market Swings")
        st.plotly_chart(fig, use_container_width=True)
        
        # High vs Low 
        fig_hl = go.Figure()
        fig_hl.add_trace(go.Scatter(x=df["Timestamp"], y=df["High"], name="High", line=dict(color="green")))
        fig_hl.add_trace(go.Scatter(x=df["Timestamp"], y=df["Low"], name="Low", line=dict(color="red")))
        st.plotly_chart(fig_hl, use_container_width=True)

elif mode == "🧮 Mathematical Simulation":
    st.sidebar.subheader("🎛️ Simulation Parameters")
    pattern = st.sidebar.selectbox("Pattern", ["Sine + Noise", "Sine Wave", "Random Jump"])
    amp = st.sidebar.slider("Amplitude (Swing Size)", 100, 5000, 1000)
    freq = st.sidebar.slider("Frequency (Cycles)", 0.5, 10.0, 2.0)
    drift = st.sidebar.slider("Drift (Long-term Slope)", -100, 100, 20)
    noise = st.sidebar.slider("Noise (Volatility)", 0, 500, 100)
    
    # Updated to support 30 days 
    sim_days = st.sidebar.slider("Simulation Days", 1, 30, 30) 
    
    df_sim = generate_mathematical_data(pattern, amp, freq, drift, noise, sim_days, selected_tz)
    
    st.subheader(f"Mathematical Model: {pattern}")
    st.latex(r"Price(t) = Base + A \cdot \sin(2\pi ft) + Drift \cdot t + \epsilon")
    
    fig_sim = px.line(df_sim, x="Timestamp", y="Price", title="Generated Price Action")
    st.plotly_chart(fig_sim, use_container_width=True)

else: # Compare Both
    # Comparison logic combining both of the above
    st.write("Comparing Real Data vs Mathematical Model (30 Day Window)")
    # ... (Similar plotting logic as above for comparison)

st.sidebar.markdown("---")
st.sidebar.info("FA-2 Project: Dashboarding and Deployment ")
