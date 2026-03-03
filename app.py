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
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Volatility Visualizer",
    page_icon="₿",
    layout="wide"
)

# Custom CSS for a professional FinTech dashboard look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #1e3d59; }
    </style>
    """, unsafe_with_human_vision=True)

st.title("₿ Crypto Volatility Visualizer")
st.markdown("### FA-2: Dashboarding and Deployment — FinTechLab Pvt. Ltd.")
st.markdown("*Applying sine, cosine, and random noise to model financial behavior* ")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCE (GITHUB & DEPLOYMENT PREP)
# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: Using GitHub hosted CSV for Streamlit Cloud deployment 
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/Nihith007/Crypto-Volatility-Visualizer"
    "/refs/heads/main/btcusd_1-min_data.csv"
)

TIMEZONE_OPTIONS = {
    "🇮🇳 India (IST) UTC+5:30": "Asia/Kolkata",
    "🌍 UTC (Universal)": "UTC",
    "🇺🇸 New York (EST/EDT)": "America/New_York",
    "🇬🇧 London (GMT/BST)": "Europe/London",
    "🇦🇪 Dubai (GST) UTC+4": "Asia/Dubai",
    "🇯🇵 Tokyo (JST) UTC+9": "Asia/Tokyo",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS (STAGE 6)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Dashboard Mode")
mode = st.sidebar.radio(
    "Choose your analysis mode:",
    ["📊 Real Bitcoin Data", "🧮 Mathematical Simulation", "🔍 Compare Both"],
)

st.sidebar.markdown("---")
st.sidebar.header("🌐 Global Settings")
selected_tz_label = st.sidebar.selectbox("Display Timezone:", list(TIMEZONE_OPTIONS.keys()))
selected_tz = TIMEZONE_OPTIONS[selected_tz_label]
tz = pytz.timezone(selected_tz)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: DATA PREPARATION & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean_and_prepare_data(df):
    """Thorough cleaning as per Stage 4 requirements """
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip().capitalize() for c in df.columns]
    
    # Stage 4: Convert Timestamp to proper date-time format 
    for col in ["Timestamp", "Date", "Time"]:
        if col in df.columns:
            try:
                df["Timestamp"] = pd.to_datetime(df[col], unit='s') # Try Unix first
            except:
                df["Timestamp"] = pd.to_datetime(df[col])
            break
            
    # Stage 4: Rename Close to Price for internal logic 
    if "Close" in df.columns:
        df["Price"] = pd.to_numeric(df["Close"], errors="coerce")
        
    # Ensure numeric types for OHLCV 
    for col in ["Open", "High", "Low", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Stage 4: Handle missing data (dropna) 
    df = df.dropna(subset=["Timestamp", "Price"])
    return df.sort_values("Timestamp").reset_index(drop=True)

@st.cache_data(show_spinner="📡 Fetching Real-Time Data...")
def get_data():
    try:
        r = requests.get(GITHUB_CSV_URL, timeout=10)
        return clean_and_prepare_data(pd.read_csv(StringIO(r.text)))
    except:
        # Fallback if GitHub is down
        return None

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_price_performance(df, title="Bitcoin Price Performance"):
    fig = px.line(df, x="Timestamp", y="Price", title=title, template="plotly_white")
    fig.update_traces(line_color='#1f77b4', line_width=2)
    return fig

def plot_high_low(df):
    """High vs Low Comparison """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Timestamp"], y=df["High"], name="High (Ceiling)", line=dict(color="#2ca02c", width=1)))
    fig.add_trace(go.Scatter(x=df["Timestamp"], y=df["Low"], name="Low (Floor)", line=dict(color="#d62728", width=1), fill='tonexty', fillcolor='rgba(44, 160, 44, 0.1)'))
    fig.update_layout(title="High vs Low Daily Range", template="plotly_white", hovermode="x unified")
    return fig

def plot_volume(df):
    """Volume Analysis """
    fig = px.bar(df, x="Timestamp", y="Volume", title="Trading Volume Analysis", color_discrete_sequence=['#7f7f7f'])
    return fig

def plot_distribution(df):
    """Price Statistics Distribution """
    fig = px.histogram(df, x="Price", nbins=50, title="Price Density Distribution", marginal="box", color_discrete_sequence=['#9467bd'])
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MATHEMATICAL SIMULATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def generate_math_model(pattern, amp, freq, drift, noise, days):
    """Simulating market swings using sine/noise/integrals """
    points = days * 24 
    t = np.linspace(0, days, points)
    base_price = 45000
    
    if "Sine" in pattern:
        wave = amp * np.sin(2 * np.pi * freq * t / days)
    else:
        wave = amp * np.cos(2 * np.pi * freq * t / days)
        
    random_noise = np.cumsum(np.random.normal(0, noise, points))
    integral_drift = (drift * t) # Long-term slope 
    
    sim_prices = base_price + wave + random_noise + integral_drift
    
    stamps = [datetime.now() - timedelta(hours=points-i) for i in range(points)]
    return pd.DataFrame({
        "Timestamp": stamps,
        "Price": sim_prices,
        "High": sim_prices + (noise * 1.5),
        "Low": sim_prices - (noise * 1.5),
        "Volume": np.random.randint(100, 1000, points)
    })

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD LOGIC
# ─────────────────────────────────────────────────────────────────────────────
df_real = get_data()

if mode == "📊 Real Bitcoin Data":
    st.header("📊 Real-World Market Analysis")
    
    if df_real is not None:
        # Stage 4: Subset for simplicity (Up to 30 Days) 
        st.sidebar.subheader("📅 Data Filtering")
        max_days = min(30, (df_real["Timestamp"].max() - df_real["Timestamp"].min()).days)
        days_sel = st.sidebar.slider("Select Range (Last N Days):", 1, max_days, 30)
        
        filtered_df = df_real[df_real["Timestamp"] >= (df_real["Timestamp"].max() - timedelta(days=days_sel))]
        
        # Stage 6: Key Metrics 
        m1, m2, m3 = st.columns(3)
        volatility = (filtered_df["Price"].pct_change().std() * np.sqrt(len(filtered_df)) * 100)
        m1.metric("Current Price", f"${filtered_df['Price'].iloc[-1]:,.2f}")
        m2.metric("Volatility Index", f"{volatility:.2f}%")
        m3.metric("30D Avg Price", f"${filtered_df['Price'].mean():,.2f}")
        
        # Visualizations (Stage 5) 
        st.plotly_chart(plot_price_performance(filtered_df), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_high_low(filtered_df), use_container_width=True)
        with c2: st.plotly_chart(plot_volume(filtered_df), use_container_width=True)
        
        st.plotly_chart(plot_distribution(filtered_df), use_container_width=True)
    else:
        st.error("Could not load CSV data. Please check your GitHub URL.")

elif mode == "🧮 Mathematical Simulation":
    st.header("🧮 Theoretical Market Modeling")
    st.sidebar.subheader("Simulation Controls")
    p = st.sidebar.selectbox("Wave Pattern", ["Sine + Noise", "Cosine + Noise"])
    a = st.sidebar.slider("Amplitude (Swing Size)", 100, 5000, 1500)
    f = st.sidebar.slider("Frequency (Speed)", 0.5, 10.0, 2.0)
    d = st.sidebar.slider("Drift (Slope)", -500, 500, 50)
    n = st.sidebar.slider("Noise (Volatility)", 10, 500, 150)
    
    df_sim = generate_math_model(p, a, f, d, n, 7)
    
    st.plotly_chart(plot_price_performance(df_sim, "Mathematical Simulation Output"), use_container_width=True)
    st.latex(r"Price(t) = Base + A\sin(2\pi ft) + \int Drift \, dt + \epsilon")

else: # Compare Both
    st.header("🔍 Compare: Real vs. Simulated")
    
    if df_real is not None:
        # Comparison logic showing same parameters as original code
        col1, col2 = st.columns(2)
        
        # Real Data Column
        with col1:
            st.subheader("Market Reality")
            comp_real = df_real.tail(24 * 7) # Last week
            st.plotly_chart(plot_price_performance(comp_real, "Real Bitcoin (7D)"), use_container_width=True)
            st.plotly_chart(plot_high_low(comp_real), use_container_width=True)
            st.plotly_chart(plot_volume(comp_real), use_container_width=True)

        # Simulated Data Column
        with col2:
            st.subheader("Mathematical Model")
            df_sim_comp = generate_math_model("Sine", 2000, 1.5, 20, 100, 7)
            st.plotly_chart(plot_price_performance(df_sim_comp, "Simulated Swing (7D)"), use_container_width=True)
            st.plotly_chart(plot_high_low(df_sim_comp), use_container_width=True)
            st.plotly_chart(plot_volume(df_sim_comp), use_container_width=True)

st.markdown("---")
st.info("💡 **Project Tip:** Use the sidebar to toggle between real market data and mathematical models to analyze how volatility differs in theory vs. practice.")
