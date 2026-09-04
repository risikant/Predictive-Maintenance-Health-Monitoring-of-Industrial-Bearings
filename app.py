import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Bearing Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# -------------------------------------------------------------
# 2. Paths & Data Caching
# -------------------------------------------------------------
DATA_FOLDER = r"C:\Users\risik\OneDrive\Desktop\Predictive Maintenance for Industrial Bearings\2nd_test"
FEATURES_CSV = r"C:\Users\risik\OneDrive\Desktop\Predictive Maintenance for Industrial Bearings\ims_bearing_features.csv"

@st.cache_data
def load_feature_data(csv_path):
    return pd.read_csv(csv_path)

@st.cache_resource
def train_models(df):
    features = ["RMS", "Kurtosis", "Peak_to_Peak", "Crest_Factor", "Skewness", "Max_FFT_Peak"]
    X = df[features]
    y_class = df["Health_State"]
    y_reg = df["RUL_Cycles"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_class)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y_reg)

    return clf, reg

@st.cache_data
def get_snapshot_files(folder_path):
    return sorted(glob.glob(os.path.join(folder_path, "*")))

# Load cached data and models
df_features = load_feature_data(FEATURES_CSV)
clf_model, reg_model = train_models(df_features)
snapshot_files = get_snapshot_files(DATA_FOLDER)

# -------------------------------------------------------------
# 3. Sidebar Controls
# -------------------------------------------------------------
st.sidebar.header("🕹️ Operational Controls")
st.sidebar.markdown("Slide through the 984 run-to-failure snapshots to observe vibration progression in real-time.")

total_cycles = len(snapshot_files)
selected_cycle = st.sidebar.slider(
    "Select Snapshot Cycle:",
    min_value=0,
    max_value=total_cycles - 1,
    value=530,
    step=1
)

selected_file_path = snapshot_files[selected_cycle]
filename = os.path.basename(selected_file_path)
st.sidebar.caption(f"**Current File:** `{filename}`")

# -------------------------------------------------------------
# 4. Load Raw Vibration & Live Metrics Calculation
# -------------------------------------------------------------
df_raw = pd.read_csv(selected_file_path, sep=r'\s+', header=None)
b1_signal = df_raw[0].values  # Bearing 1 Channel
sampling_rate = 20480
time_axis = np.linspace(0, len(b1_signal) / sampling_rate, len(b1_signal))

# Compute Live Metrics
live_rms = np.sqrt(np.mean(b1_signal**2))
live_kurtosis = kurtosis(b1_signal)
live_ptp = np.ptp(b1_signal)
live_crest = np.max(np.abs(b1_signal)) / (live_rms + 1e-8)
live_skewness = skew(b1_signal)

# FFT Calculation
n_samples = len(b1_signal)
freq_axis = rfftfreq(n_samples, d=1.0 / sampling_rate)
fft_amplitudes = np.abs(rfft(b1_signal)) * (2.0 / n_samples)
live_max_fft = np.max(fft_amplitudes)

# Live Model Inference
live_feature_vector = pd.DataFrame([{
    "RMS": live_rms,
    "Kurtosis": live_kurtosis,
    "Peak_to_Peak": live_ptp,
    "Crest_Factor": live_crest,
    "Skewness": live_skewness,
    "Max_FFT_Peak": live_max_fft
}])

predicted_state = clf_model.predict(live_feature_vector)[0]
predicted_rul = reg_model.predict(live_feature_vector)[0]

# -------------------------------------------------------------
# 5. Dashboard Header & KPI Banners
# -------------------------------------------------------------
st.title("⚙️ Industrial Bearing Predictive Maintenance")
st.markdown("Real-time vibration signal monitoring, health diagnosis, and RUL estimation on the **NASA IMS Dataset**.")

col1, col2, col3, col4 = st.columns(4)

# Color badge styling for status
if predicted_state == "Normal":
    state_color = "green"
elif predicted_state == "Early_Degradation":
    state_color = "orange"
else:
    state_color = "red"

col1.metric("Predicted Health State", f":{state_color}[{predicted_state}]")
col2.metric("Predicted RUL", f"{predicted_rul:.1f} cycles", f"{predicted_rul * 10 / 60:.1f} hours")
col3.metric("Live RMS", f"{live_rms:.4f} g")
col4.metric("Live Kurtosis", f"{live_kurtosis:.2f}")

st.divider()

# -------------------------------------------------------------
# 6. Live Waveform & Frequency Spectrum Plots
# -------------------------------------------------------------
left_plot_col, right_plot_col = st.columns(2)

with left_plot_col:
    st.subheader("📈 Time-Domain Acceleration Waveform")
    # Subsample for rendering performance
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=time_axis[::4], 
        y=b1_signal[::4],
        mode="lines",
        line=dict(color="#1f77b4", width=1),
        name="Bearing 1 (g)"
    ))
    fig_time.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Acceleration (g)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_time, use_container_width=True)

with right_plot_col:
    st.subheader("📊 Frequency Spectrum (FFT)")
    # Zoom in to 0 - 2000 Hz to capture rotational & fault harmonics
    freq_mask = freq_axis <= 2000
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(
        x=freq_axis[freq_mask], 
        y=fft_amplitudes[freq_mask],
        mode="lines",
        line=dict(color="#d62728", width=1.2),
        name="Amplitude Spectrum"
    ))
    # Shaft speed (2000 RPM ≈ 33.3 Hz) and BPFO (~236 Hz) reference markers
    fig_fft.add_vline(x=33.33, line_dash="dash", line_color="gray", annotation_text="1X RPM")
    fig_fft.add_vline(x=236.4, line_dash="dash", line_color="orange", annotation_text="BPFO (Outer Race)")
    fig_fft.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_fft, use_container_width=True)

# -------------------------------------------------------------
# 7. Whole Lifecycle Trajectory Indicator
# -------------------------------------------------------------
st.subheader("📍 Lifetime Degradation Trajectory")

fig_traj = go.Figure()

# RMS baseline line
fig_traj.add_trace(go.Scatter(
    x=df_features["Cycle"], 
    y=df_features["RMS"],
    mode="lines",
    line=dict(color="navy", width=1.5),
    name="Historical RMS (Energy)"
))

# Current operating marker
fig_traj.add_trace(go.Scatter(
    x=[selected_cycle],
    y=[live_rms],
    mode="markers",
    marker=dict(size=12, color="red", symbol="circle"),
    name="Current Position"
))

# Threshold markings
fig_traj.add_vline(x=530, line_dash="dot", line_color="orange", annotation_text="Warning (Cycle 530)")
fig_traj.add_vline(x=700, line_dash="dot", line_color="red", annotation_text="Severe Fault (Cycle 700)")

fig_traj.update_layout(
    xaxis_title="Cycle (10-minute snapshot increments)",
    yaxis_title="RMS Acceleration (g)",
    height=320,
    margin=dict(l=20, r=20, t=30, b=20)
)
st.plotly_chart(fig_traj, use_container_width=True)