from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
import multiprocessing
import os
import sys
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

# =======================================================
# FIX FOR WINDOWS ONLY                                  #
# =======================================================
os.environ['HADOOP_HOME'] = "C:\\hadoop"                #
sys.path.append("C:\\hadoop\\bin")                      #
os.environ['PATH'] += os.pathsep + "C:\\hadoop\\bin"    #
# =======================================================

plots_path = 'PLOTS/03-TEMPORAL ANALYSIS'
os.makedirs(plots_path, exist_ok=True)
sns.set_theme(style="darkgrid")

cores = multiprocessing.cpu_count()
spark = (
    SparkSession.builder.appName("Spark")
    .master("local[*]")
    # RAM
    .config("spark.driver.memory", "8g")
    # Set partitions to 2 times the number of the cores
    .config("spark.sql.shuffle.partitions", str(cores * 2))
    .config("spark.default.parallelism", str(cores * 2))
    # serializer
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    #.config("spark.ui.showConsoleProgress", "true")
    .getOrCreate()
)

sc = spark.sparkContext
df_raw = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("DATA/downloads/*.csv") \
    .withColumn("source_file", input_file_name())


# ===========================================================================================
# Downsampling
# ===========================================================================================
new_column_names = [
    "Time", "CO_ppm", "Humidity", "Temperature", "Flow_rate", "Heater_voltage",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", 
    "R8", "R9", "R10", "R11", "R12", "R13", "R14", 
    "source_file"
]
df_renamed = df_raw.toDF(*new_column_names)

df_bucketed = df_renamed.withColumn("Time", F.floor(F.col("Time") / 300))

cols_to_avg = [c for c in df_renamed.columns if c not in ["Time","source_file"]]
calc_mean = [F.mean(c).alias(c) for c in cols_to_avg]

df_grouped_by_file = df_bucketed.groupBy("source_file", "Time").agg(*calc_mean)
df_sorted = df_grouped_by_file.orderBy("source_file", "Time")

start_date = "2025-09-01 00:00:00"
w = Window.orderBy("source_file", "Time")
df_final = df_sorted.withColumn("Timestamp",F.to_timestamp(F.lit(start_date)) + (F.row_number().over(w) - 1) * F.expr("INTERVAL 5 MINUTES")).drop("Time","source_file")


data = df_final.toPandas().set_index('Timestamp')

sensors = ["R" + str(i) for i in range(1,15)]
data["Sensors_Mean"] = data[sensors].mean(axis=1)

# =======================================
# TEMPORAL ANALYSIS
# =======================================
# Seasonal decomposition
# Finding out the number of periods

decomp = seasonal_decompose(data["Sensors_Mean"], model="additive", period=300)

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

# Component‑strength metrics (trend & seasonality)
# Drop NaNs that appear at the edges of trend/residual series
trend = decomp.trend.dropna()
seasonal = decomp.seasonal.dropna()
resid = decomp.resid.dropna()

trend_strength = 1 - np.var(resid) / np.var(trend + resid)
season_strength = 1 - np.var(resid) / np.var(seasonal + resid)

print("==== Additive Seasonal Decomposition ====")
print(f"Trend strength      : {trend_strength:.3f}")
print(f"Seasonality strength: {season_strength:.3f}")

# STL (Seasonal Trend Decomposition using LOESS) decomposition - robust to outliers
stl = STL(data["Sensors_Mean"], period=300, robust=True)
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

# Component‑strength metrics (trend & seasonality)
# Drop NaNs that appear at the edges of trend/residual series
trend = stl_res.trend.dropna()
seasonal = stl_res.seasonal.dropna()
resid = stl_res.resid.dropna()

trend_strength = 1 - np.var(resid) / np.var(trend + resid)
season_strength = 1 - np.var(resid) / np.var(seasonal + resid)

print("==== STL Decomposition ====")
print(f"Trend strength      : {trend_strength:.3f}")
print(f"Seasonality strength: {season_strength:.3f}")

# ACF & PACF of the residuals (post‑decomposition check)
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


# Stationarity tests – ADF & KPSS

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


# Run tests on the raw series
adf_report(data["Sensors_Mean"])
kpss_report(data["Sensors_Mean"], regression="c")  # constant only (no trend)