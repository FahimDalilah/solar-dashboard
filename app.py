import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.set_page_config(page_title="Solar Energy Forecasting", layout="wide")

DATA_PATH  = "data/engineered_solar_data_dashboard.csv"

TARGET_COL = "SolarGeneration"

def create_sequences(X: np.ndarray, y: np.ndarray, window: int):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window, :])  # <-- explicit 2D slice
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

@st.cache_resource
def load_lgb_model():
    return joblib.load("models/best_lightgbm_hybrid_tuned.pkl")

@st.cache_resource
def load_feature_cols():
    return joblib.load("models/feature_cols.pkl")

@st.cache_resource
def load_tcn_window():
    return int(joblib.load("models/tcn_window.pkl"))

@st.cache_resource
def load_tcn_scaler():
    return joblib.load("models/tcn_scaler.pkl")

@st.cache_resource
def load_tcn():
    return load_model("models/best_tcn_hybrid.h5")

    lgb_model = load_lgb_model()
    tcn_model = load_model("models/best_tcn_hybrid.h5")
    scaler = joblib.load("models/tcn_scaler.pkl")
    FEATURE_COLS = load_feature_cols()
    SEQ_LENGTH = joblib.load("models/tcn_window.pkl")

    return lgb_model, tcn_model, scaler, feature_cols, SEQ_LENGTH

@st.cache_data
def load_data():
    df = pd.read_csv("data/engineered_solar_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df

st.title("🌞 Solar Energy Forecasting Dashboard")
st.markdown("**Hybrid Feature Engineering | LightGBM vs TCN**")

try:
    lgb_model, tcn_model, scaler, FEATURE_COLS, SEQ_LENGTH
except Exception as e:
    st.error(str(e))
    st.stop()

df = load_data()

# sidebar controls (same as your doc)
st.sidebar.header("Settings")
split_ratio = st.sidebar.slider("Train/Test Split", 0.6, 0.9, 0.8)
forecast_horizon = st.sidebar.slider("Forecast Horizon (steps)", 24, 168, 72)
plot_points = st.sidebar.slider("Plot points", 200, 5000, 800, step=100)

# prepare X/y
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    st.error(f"Engineered dataset is missing feature columns: {missing_cols}. Re-run 00_prepare_data.py")
    st.stop()

X = df[FEATURE_COLS]
y = df[TARGET_COL].values
ts = df["Timestamp"].values

split_idx = int(len(X) * split_ratio)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
ts_test = ts[split_idx:]

# ---- LightGBM preds ----
X = df[FEATURE_COLS]
lgb_preds = lgb_model.predict(X)

st.success("LightGBM predictions generated successfully")

# ---- TCN preds (IMPORTANT: transform only) ----
# ---------- TCN predictions (SAFE SHAPES) ----------
X_scaled = scaler.transform(X.values)
y_all = y  # already numpy

# build sequences over the FULL test region
X_test_scaled = X_scaled[split_idx:]
y_test_part = y[split_idx:]

# include history so first window is valid
X_hist = X_scaled[max(0, split_idx - SEQ_LENGTH):split_idx]
y_hist = y[max(0, split_idx - SEQ_LENGTH):split_idx]

X_concat = np.vstack([X_hist, X_test_scaled])
y_concat = np.concatenate([y_hist, y_test_part])

X_seq, y_seq = create_sequences(X_concat, y_concat, SEQ_LENGTH)

# HARD CHECK (VERY IMPORTANT)
assert X_seq.ndim == 3, f"X_seq must be 3D, got {X_seq.shape}"
assert X_seq.shape[1] == SEQ_LENGTH, f"Wrong window size: {X_seq.shape}"
assert X_seq.shape[2] == X.shape[1], f"Feature mismatch: {X_seq.shape}"

tcn_preds = tcn_model.predict(X_seq, verbose=0).flatten()

# align timestamps
# ---------- Align TCN timestamps ----------
ts_tcn = ts_test[SEQ_LENGTH:]

pl_lgb = min(plot_points, len(ts_test))
pl_tcn = min(plot_points, len(ts_tcn), len(tcn_preds))

st.subheader("Evaluation Metrics Comparison (LightGBM vs TCN)")

SEQ_LENGTH = 24

# ---------- Safety: convert to numpy ----------
y_lgb = np.asarray(y_test).reshape(-1)
pred_lgb = np.asarray(lgb_preds).reshape(-1)

# ---------- Align TCN with LightGBM test timeline ----------
# TCN predicts starting after SEQ_LENGTH steps, so compare on overlapping region
tcn = np.asarray(tcn_preds).reshape(-1)

# Overlap length
n_overlap = min(len(y_lgb) - SEQ_LENGTH, len(tcn), len(pred_lgb) - SEQ_LENGTH)
if n_overlap <= 0:
    st.error("Not enough overlapping points to compare LightGBM and TCN. Check your test split and SEQ_LENGTH.")
else:
    # Actuals for comparison region (starting at SEQ_LENGTH)
    y_cmp = y_lgb[SEQ_LENGTH:SEQ_LENGTH + n_overlap]

    # LightGBM preds aligned to same region
    lgb_cmp = pred_lgb[SEQ_LENGTH:SEQ_LENGTH + n_overlap]

    # TCN preds for same region
    tcn_cmp = tcn[:n_overlap]

    # ---------- Metrics ----------
    def calc_metrics(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return rmse, mae, r2

    rmse_lgb, mae_lgb, r2_lgb = calc_metrics(y_cmp, lgb_cmp)
    rmse_tcn, mae_tcn, r2_tcn = calc_metrics(y_cmp, tcn_cmp)

    # ---------- Display table ----------
    metrics_df = pd.DataFrame({
        "Model": ["LightGBM", "TCN"],
        "RMSE": [rmse_lgb, rmse_tcn],
        "MAE": [mae_lgb, mae_tcn],
        "R²": [r2_lgb, r2_tcn],
        "Compared Points": [n_overlap, n_overlap]
    })

    st.dataframe(metrics_df, use_container_width=True)

    # Optional: highlight best
    best_rmse = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
    best_mae = metrics_df.loc[metrics_df["MAE"].idxmin(), "Model"]
    best_r2 = metrics_df.loc[metrics_df["R²"].idxmax(), "Model"]

    st.caption(
        f"Best RMSE: {best_rmse} | Best MAE: {best_mae} | Best R²: {best_r2}. "
        f"Comparison aligned using SEQ_LENGTH = {SEQ_LENGTH}."
    )

# ======================
# Predicted vs Actual
# ======================
st.subheader("Actual vs Predicted (LightGBM + TCN)")

SEQ_LENGTH = 24

# How many points to show (use your slider if you want)
N = min(plot_points, 500, len(y_test), len(lgb_preds))

steps = np.arange(N)

# TCN starts at step 24
# We can only plot TCN where it overlaps inside the first N steps
tcn_max = min(len(tcn_preds), max(0, N - SEQ_LENGTH))

fig, ax = plt.subplots(figsize=(14, 5))

# ----- ONE actual line -----
ax.plot(steps, y_test[:N], label="Actual", linewidth=2, color="black")

# ----- LightGBM predicted -----
ax.plot(steps, lgb_preds[:N], label="Predicted (LightGBM)", alpha=0.85)

# ----- TCN predicted (shifted right by 24) -----
if tcn_max > 10:
    steps_tcn = np.arange(SEQ_LENGTH, SEQ_LENGTH + tcn_max)
    ax.plot(steps_tcn, tcn_preds[:tcn_max], label="Predicted (TCN)", alpha=0.85)

ax.set_title("Actual vs Predicted (LightGBM + TCN)")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Solar Generation")
ax.legend()
ax.grid(True)

st.pyplot(fig)
st.caption("Note: TCN predictions start from step 24 because the first 24 steps are used as the input window (sequence length).")

# ======================
# Error Distribution
# ======================
st.subheader("Error Distribution")
lgb_err = y_test - lgb_preds
tcn_err = y_seq - tcn_preds

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(lgb_err, bins=50, alpha=0.6, label="LightGBM")
ax.hist(tcn_err, bins=50, alpha=0.6, label="TCN")
ax.set_xlabel("Prediction Error")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# ======================
# Forecast (LightGBM recursive)
# ======================
st.subheader("Forecasting Graph (History + Next Timestamps)")

# ----------------------------
# 1) Prepare clean hourly history
# ----------------------------
df_plot = df.copy()
df_plot["Timestamp"] = pd.to_datetime(df_plot["Timestamp"])
df_plot = df_plot.sort_values("Timestamp")

# Ensure numeric
df_plot[TARGET_COL] = pd.to_numeric(df_plot[TARGET_COL], errors="coerce")
df_plot = df_plot.dropna(subset=[TARGET_COL])

# 🔥 IMPORTANT FIX: SUM, not MEAN
df_plot = (
    df_plot.set_index("Timestamp")
    .resample("1H")[TARGET_COL]
    .sum()
    .reset_index()
)

# Use enough history to show cycles
df_plot = df_plot.tail(24 * 7).copy()  # 7 days

# aggregate duplicates safely
df_plot = (
    df_plot.groupby("Timestamp", as_index=False)
    .mean(numeric_only=True)
    .sort_values("Timestamp")
)

# ----------------------------
# 2) Build hourly solar profile
# ----------------------------
df_plot["hour"] = df_plot["Timestamp"].dt.hour

hourly_profile = (
    df_plot.groupby("hour")[TARGET_COL]
    .mean()
    .reindex(range(24))
    .fillna(0.0)
    .rolling(window=5, center=True, min_periods=1)
    .mean()
)

# Fill night hours with 0
hourly_profile = hourly_profile.fillna(0.0)

# Smooth the curve (no math errors)
hourly_profile = (
    hourly_profile
    .rolling(window=3, center=True, min_periods=1)
    .mean()
)

hourly_profile = hourly_profile.rolling(window=5, center=True, min_periods=1).mean()

# Hour-of-day statistics (mean & std) for confidence bands
df_plot["hour"] = df_plot["Timestamp"].dt.hour

hourly_stats = df_plot.groupby("hour")[TARGET_COL].agg(["mean", "std"]).reindex(range(24))
hourly_stats["mean"] = hourly_stats["mean"].fillna(0.0)
hourly_stats["std"]  = hourly_stats["std"].fillna(0.0)

# Smooth mean/std a bit to avoid spiky bands (optional but nicer)
hourly_stats["mean"] = hourly_stats["mean"].rolling(window=5, center=True, min_periods=1).mean()
hourly_stats["std"]  = hourly_stats["std"].rolling(window=5, center=True, min_periods=1).mean()

# Confidence band multiplier: ~1.0 (wide-ish) or 1.96 (~95% if normal-ish)
k = st.slider("Confidence band width (k × std)", 0.0, 3.0, 1.0, step=0.1)

# Shift everything to start at today (visual only)
#shift = pd.Timestamp.today().normalize() - df_plot["Timestamp"].iloc[-1]

#df_plot["Timestamp"] += shift
#future_ts = [ts + shift for ts in future_ts]

# ----------------------------
# 3) Generate future timestamps
# ----------------------------
last_ts = df_plot["Timestamp"].iloc[-1]
future_ts = [last_ts + pd.Timedelta(hours=i+1) for i in range(forecast_horizon)]

future_mean  = [float(hourly_stats.loc[ts.hour, "mean"]) for ts in future_ts]
future_std   = [float(hourly_stats.loc[ts.hour, "std"])  for ts in future_ts]
future_lower = [max(0.0, m - k*s) for m, s in zip(future_mean, future_std)]
future_upper = [m + k*s for m, s in zip(future_mean, future_std)]

sunrise = st.slider("Sunrise hour", 0, 12, 7)
sunset  = st.slider("Sunset hour", 12, 23, 19)
show_labels = st.checkbox("Label sunrise/sunset", value=False)

# ----------------------------
# 4) Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(14, 5))

# History
ax.plot(
    df_plot["Timestamp"],
    df_plot[TARGET_COL],
    label="History (Hourly Actual)",
    color="black",
    linewidth=2
)

# Forecast mean
ax.plot(
    future_ts,
    future_mean,
    label="Forecast (Hourly Profile)",
    linewidth=2,
    alpha=0.9
)

# Confidence band
ax.fill_between(
    future_ts,
    future_lower,
    future_upper,
    alpha=0.2,
    label=f"Confidence Band (± {k:.1f}×std)"
)

# Split marker
ax.axvline(last_ts, linestyle="--", alpha=0.7)

# ---- Sunrise/Sunset shading (night periods) ----
# Determine day range covering history+forecast
t_start = min(df_plot["Timestamp"].iloc[0], future_ts[0])
t_end   = max(df_plot["Timestamp"].iloc[-1], future_ts[-1])

# Build day list (midnight anchors)
day_start = pd.Timestamp(t_start.date())
day_end   = pd.Timestamp(t_end.date())

days = pd.date_range(day_start, day_end, freq="D")

for d in days:
    # Night: 00:00 -> sunrise
    night1_start = d
    night1_end = d + pd.Timedelta(hours=sunrise)
    ax.axvspan(night1_start, night1_end, alpha=0.08)

    # Night: sunset -> 24:00
    night2_start = d + pd.Timedelta(hours=sunset)
    night2_end = d + pd.Timedelta(days=1)
    ax.axvspan(night2_start, night2_end, alpha=0.08)

    if show_labels:
        ax.axvline(d + pd.Timedelta(hours=sunrise), linestyle=":", alpha=0.35)
        ax.axvline(d + pd.Timedelta(hours=sunset), linestyle=":", alpha=0.35)
        ax.text(d + pd.Timedelta(hours=sunrise), ax.get_ylim()[1]*0.98, "Sunrise",
                rotation=90, va="top", ha="right", alpha=0.6)
        ax.text(d + pd.Timedelta(hours=sunset), ax.get_ylim()[1]*0.98, "Sunset",
                rotation=90, va="top", ha="right", alpha=0.6)

ax.set_title("History + Forecast (Hourly Solar Profile) with Confidence Band")
ax.set_xlabel("Time")
ax.set_ylabel("Solar Generation")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.caption(
    "Confidence band is derived from historical variability by hour-of-day (mean ± k×std). "
    "Blue shaded regions indicate day-time based on sunrise/sunset hours."
)

# ==============================
# 9. FOOTER
# ==============================
st.markdown("---")
st.caption("Final Year Project – Solar Energy Forecasting using Hybrid Feature Engineering")
