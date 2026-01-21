import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

plots_path = 'PLOTS/03-TEMPORAL ANALYSIS'
os.makedirs(plots_path, exist_ok=True)
sns.set_theme(style="darkgrid")

data = pd.read_csv("DATA/log_data_Downsampled.csv", index_col='Timestamp', parse_dates=True)

# ===========================================================================================
# TEMPORAL ANALYSIS
# ===========================================================================================

# ===========================================================================================
# Seasonal decomposition
# ===========================================================================================
# Finding out the number of periods

decomp = seasonal_decompose(data["Sensors_Mean"], model="additive", period=13)

# Plot the four components
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
axes[0].plot(data["Sensors_Mean"], label="Observed", color="steelblue")
axes[0].legend(loc="upper left")

axes[1].plot(decomp.trend, label="Trend", color="darkorange")
axes[1].legend(loc="upper left")

axes[2].plot(decomp.seasonal, label="Seasonal", color="green")
axes[2].legend(loc="upper left")

axes[3].plot(decomp.resid, label="Residual", color="crimson")
axes[3].legend(loc="upper left")
axes[3].set_xlabel("Date")

fig.suptitle("Additive Seasonal Decomposition", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(plots_path, 'Additive_Seasonal_Decomposition.png'), dpi=300)
plt.close()

# ===========================================================================================
# Component‑strength metrics (trend & seasonality)
# ===========================================================================================
# Drop NaNs that appear at the edges of trend/residual series
trend = decomp.trend.dropna()
seasonal = decomp.seasonal.dropna()
resid = decomp.resid.dropna()

trend_strength = 1 - np.var(resid) / np.var(trend + resid)
season_strength = 1 - np.var(resid) / np.var(seasonal + resid)

print("==== Additive Seasonal Decomposition ====")
print(f"Trend strength      : {trend_strength:.3f}")
print(f"Seasonality strength: {season_strength:.3f}")

# ===========================================================================================
# STL (Seasonal Trend Decomposition using LOESS) decomposition - robust to outliers
# ===========================================================================================
stl = STL(data["Sensors_Mean"], period=13, robust=True)
stl_res = stl.fit()

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
axes[0].plot(stl_res.observed, label="Observed")
axes[0].legend()

axes[1].plot(stl_res.trend, label="Trend", color="darkorange")
axes[1].legend()

axes[2].plot(stl_res.seasonal, label="Seasonal", color="green")
axes[2].legend()

axes[3].plot(stl_res.resid, label="Residual", color="crimson")
axes[3].set_xlabel("Date")
axes[3].legend()

plt.suptitle("STL Decomposition (robust)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(plots_path, 'STL_Decomposition.png'), dpi=300)
plt.close()

# ===========================================================================================
# Component‑strength metrics (trend & seasonality)
# ===========================================================================================
# Drop NaNs that appear at the edges of trend/residual series
trend = stl_res.trend.dropna()
seasonal = stl_res.seasonal.dropna()
resid = stl_res.resid.dropna()

trend_strength = 1 - np.var(resid) / np.var(trend + resid)
season_strength = 1 - np.var(resid) / np.var(seasonal + resid)

print("==== STL Decomposition ====")
print(f"Trend strength      : {trend_strength:.3f}")
print(f"Seasonality strength: {season_strength:.3f}")

# ===========================================================================================
# ACF & PACF of the residuals (post‑decomposition check)
# ===========================================================================================
fig, ax = plt.subplots(3, 1, figsize=(10, 9))
n_lags = 72 # 6 hours
plot_acf(resid, lags=n_lags, ax=ax[0], title="ACF – Residuals")
plot_pacf(resid, lags=n_lags, ax=ax[1], title="PACF – Residuals")

# Quick visual: residuals should look like white noise
sns.lineplot(data=resid, ax=ax[2])
ax[2].set_title("Residuals")
ax[2].set_ylabel("Residual")

plt.tight_layout()
plt.savefig(os.path.join(plots_path, 'ACF_&_PACF_of_the_residuals.png'), dpi=300)
plt.close()

# ===========================================================================================
# Stationarity tests – ADF & KPSS
# ===========================================================================================
def adf_report(series):
    result = adfuller(series, autolag="AIC")
    print("\n=== Augmented Dickey‑Fuller Test ===")
    print(f"ADF Statistic   : {result[0]:.4f}")
    print(f"P‑value         : {result[1]:.4f}")
    print("Critical values :")
    for key, val in result[4].items():
        print(f"   {key} : {val:.4f}")
    if result[1] < 0.05:
        print("=> Reject H0 – the series is stationary.")
    else:
        print("=> Fail to reject H0 – the series is non‑stationary.")

def kpss_report(series, regression="c"):
    result = kpss(series, regression=regression, nlags="auto")
    print("\n=== KPSS Test ===")
    print(f"KPSS Statistic  : {result[0]:.4f}")
    print(f"P‑value         : {result[1]:.4f}")
    print("Critical values :")
    for key, val in result[3].items():
        print(f"   {key} : {val:.4f}")
    if result[1] < 0.05:
        print("=> Reject H0 – the series is NOT stationary (has a unit root).")
    else:
        print("=> Fail to reject H0 – the series is stationary.")

# ===========================================================================================
# Run tests on the raw series
# ===========================================================================================
print(data.columns)

adf_report(data["Sensors_Mean"])
kpss_report(data["Sensors_Mean"], regression="c")  # constant only (no trend)
