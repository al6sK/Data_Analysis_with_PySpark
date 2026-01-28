from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
import multiprocessing
import os
import sys
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, mean, variance, lit, max,format_number,avg,window,round
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =======================================================
# FIX FOR WINDOWS ONLY                                  #
# =======================================================
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'              #
os.environ['SPARK_DRIVER_HOST'] = '127.0.0.1'           #
os.environ['HADOOP_HOME'] = "C:\\hadoop"                #
sys.path.append("C:\\hadoop\\bin")                      #
os.environ['PATH'] += os.pathsep + "C:\\hadoop\\bin"    #
# =======================================================

plots_path = 'PLOTS/01-PREPROCESSIN-AND-DATA-UNDERSTANDING'
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

print(f"rows: {df_raw.count()}")
# df_raw.printSchema()

new_column_names = [
    "Time", "CO_ppm", "Humidity", "Temperature", "Flow_rate", "Heater_voltage",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", 
    "R8", "R9", "R10", "R11", "R12", "R13", "R14", 
    "source_file"
]
df_renamed = df_raw.toDF(*new_column_names)

# ===========================================================================================
# Downsampling to 5 min
# ===========================================================================================
df_bucketed = df_renamed.withColumn("Time", F.floor(F.col("Time") / 300))

cols_to_avg = [c for c in df_bucketed.columns if c not in ["Time","source_file"]]
calc_avg = [F.mean(c).alias(c) for c in cols_to_avg]

df_grouped_by_file = df_bucketed.groupBy("source_file", "Time").agg(*calc_avg)
df_sorted = df_grouped_by_file.orderBy("source_file", "Time")

df_sorted.select("Time", "source_file","CO_ppm", "Humidity", "Temperature", "Flow_rate", "Heater_voltage", "R14").show(10, truncate=False)
print(f"rows: {df_sorted.count()}")

start_date = "2025-09-01 00:00:00"
w = Window.orderBy("source_file", "Time")
df_final = df_sorted.withColumn(
    "Timestamp",
    F.to_timestamp(F.lit(start_date)) + (F.row_number().over(w) - 1) * F.expr("INTERVAL 5 MINUTES")
    ).drop("Time","source_file")

df_final.select("Timestamp","CO_ppm", "Humidity", "Temperature", "Flow_rate", "Heater_voltage", "R14").show(10, truncate=False)
print(f"Final rows: {df_final.count()}")

# ===========================================================================================
# Preprocessing
# ===========================================================================================

# Find null values 
df_final.select([count(when(col(c).isNull(), 1)).alias(c) for c in df_final.columns]).show()

data = df_final.toPandas().set_index('Timestamp')
print(data.shape)
print(data.head())

# save to csv file for later
data.to_csv("DATA/raw_data_Downsampled.csv", index=True)


# ===========================================================================================
# Box plot
# ===========================================================================================
def boxplot(data,filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    data.select_dtypes(include=["number"]).plot.box(
        ax=ax,
        rot=30,  # Rotate labels for readability
        showmeans=True,  # Show mean indicator
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
        patch_artist=True,  # Fill boxes with color
    )

    ax.set_yscale("symlog", linthresh=10)

    # Grid & Layout
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=-0.05, top=None)

    # Labels and Title
    ax.set_ylabel("Count (Log Scale)", fontsize=12)
    ax.set_title(
        "Distribution of Features",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=30,
    )
    plt.figtext(
        0.1,
        0.92,
        "Box plot representing statistics with log scaling",
        ha="left",
        fontsize=10,
    )

    # Seaborn Style
    sns.despine()
    plt.tight_layout()

    plt.savefig(os.path.join(plots_path, f'{filename}.png'), dpi=300, bbox_inches="tight")
    plt.close()

boxplot(data,"Box_Plot")
# ===========================================================================================
# Distribution_histplots
# ===========================================================================================
Distribution_histplots_path = os.path.join(plots_path, 'Distribution_histplots')
os.makedirs(Distribution_histplots_path, exist_ok=True)

numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
palette = sns.color_palette("deep")
for i in range(len(numeric_cols)):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Custom color for better visibility
    color = palette[i % len(palette)]

    sns.histplot(
        data=data,
        x=numeric_cols[i],
        bins=50,
        kde=True,
        ax=ax,
        color=color,
        alpha=0.6,
    )
    # Set labels with improved clarity
    ax.set_xlabel(numeric_cols[i], fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    # Adding skewness annotation
    skewness = data[numeric_cols[i]].skew()
    ax.text(
        0.95,
        0.85,
        f"Skewness: {skewness:.2f}",
        transform=ax.transAxes,
        ha="right",
        color="black",
        weight="bold",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )
    # Add title and subtitle
    ax.set_title(
        numeric_cols[i]+" Distribution",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    plt.figtext(
        0.1,
        0.9,
        "Histogram with KDE overlay",
        fontsize=10,
        ha="left",
    )

    # Grid and despine for a cleaner look
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    sns.despine(left=True)
    plt.tight_layout()

    plt.savefig(os.path.join(Distribution_histplots_path, numeric_cols[i] + "_Distribution_histplot.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ===========================================================================================
# Logarithmic Transformation on positive skew features 
# ===========================================================================================

log_cols = [f'R{i}' for i in range(1, 15)]
data[log_cols] = np.log1p(data[log_cols])

boxplot(data,"Box_Plot_log_data")

# save to csv file for later
data.to_csv("DATA/log_data_Downsampled.csv", index=True)

# ===========================================================================================
# Correlation Matrix
# ===========================================================================================
corr = data.select_dtypes(include=["number"]).corr(method="pearson").round(2)

# Create the mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Create the heatmap with the mask
plt.figure(figsize=(10, 10))
sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    vmin=-1,
    vmax=1,  # Ensure that color scaling is consistent
    cbar_kws={"label": "Correlation Coefficient"},
    annot_kws={"size": 10},  # Adjust annotation size
    mask=mask,  # Apply the mask to hide the upper triangle
)

# Title and labels for context
plt.title("Correlation Matrix of Numerical Features", fontsize=16, fontweight="bold")
plt.xlabel("Features", fontsize=12)
plt.ylabel("Features", fontsize=12)

# Rotate the axis labels for better readability
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(plots_path, 'Correlation_Plot.png'), dpi=300, bbox_inches="tight")
plt.close()

# ===========================================================================================
# Plot CO_ppm","R1","R7","R14 values over time
# ===========================================================================================
columns_for_norm = ["CO_ppm","R1","R7","R14"]
for feature in columns_for_norm:
    min_val = data[feature].min()
    max_val = data[feature].max()
    denominator = max_val - min_val
    if denominator == 0:
        data[feature] = 0.0
    else:
        data[feature] = (data[feature] - min_val) / denominator

sampled_data=data[:25]
colors = sns.color_palette("bright", 5)
# plot CO_ppm_vs_R1_R7_R14
plt.figure(figsize=(15,8))
plt.plot(sampled_data['CO_ppm'], label='CO_ppm', color=colors[0], linewidth=1.5,alpha=0.8)
plt.plot(sampled_data['R1'], label='R1', color=colors[1], linewidth=1.5,alpha=0.8)
plt.plot(sampled_data['R7'], label='R7', color=colors[2], linewidth=1.5,alpha=0.8)
plt.plot(sampled_data['R14'], label='R14', color=colors[3], linewidth=1.5,alpha=0.8)
plt.title('CO_ppm vs R1,R7,R14')
plt.xlabel('Time')
plt.ylabel('Value(normalized 0-1)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid("x")
plt.tight_layout()
plt.savefig(os.path.join(plots_path, 'CO_ppm_vs_R1_R7_R14.png'), dpi=300)
plt.close()

# ===========================================================================================
# (i) Sum of CO_ppm in all record that have Heater_voltage >= Heater_voltage * 0.6
# ===========================================================================================
max_voltage = df_final.agg(F.max("Heater_voltage")).collect()[0][0]
result = df_final.filter(F.col("Heater_voltage") > max_voltage * 0.6).agg(F.sum("CO_ppm")).collect()[0][0]
print(f"Sum of CO_ppm: {result}")

# ===========================================================================================
# (ii) Return mean and variance of each sensor values
# ===========================================================================================
sensors = ["R" + str(i) for i in range(1,15)]

mean_exprs = [F.round(mean(r),2).alias(r) for r in sensors]
df_mean = df_final.select(mean_exprs).withColumn("Statistic", lit("Mean"))
variance_exprs = [F.round(variance(r),2).alias(r) for r in sensors]
df_variance = df_final.select(variance_exprs).withColumn("Statistic", lit("Variance"))

columns = ["Statistic"] + sensors
result = df_mean.union(df_variance).select(columns)
result.show(truncate=True)

# ===========================================================================================
# (iii) Return the count of rows that have CO_ppm > 10 ppm and Humidity < 30 % RH 
# ===========================================================================================
accumulator = sc.accumulator(0) 

def count_condition(row):
    if row['CO_ppm'] > 10 and row['Humidity'] < 30:
        accumulator.add(1) 

df_final.foreach(count_condition)
print(f"{accumulator .value} is the number of rows that have CO_ppm > 10 ppm and Humidity < 30 % RH")
# ===========================================================================================
# (iv) Create a new column R, with R = ((5-V)/V) * 1_000_000 
# ===========================================================================================
df_final = df_final.withColumn("R", (( 5 - col("Heater_voltage")) / col("Heater_voltage")) * 1_000_000)
result = df_final.groupBy(F.round("Heater_voltage",2).alias("Heater_voltage")).agg(max("R").alias("max_R"))
result = result.withColumn("max_R",format_number("max_R", 2)).orderBy("Heater_voltage")
result.show(truncate=False)

# ===========================================================================================
# (v) Downsampling to 25 min
# ===========================================================================================

cols_to_avg = ["R7","CO_ppm"]

aggs = [F.mean(c).alias(c) for c in cols_to_avg]
sampled_data = df_final.groupBy(F.window("Timestamp", "25 minutes")).agg(*aggs)
# keep only the start of the returned window
sampled_data = sampled_data.withColumn("Timestamp", F.col("window.start")).drop("window").orderBy("Timestamp")
sampled_data.select("Timestamp","R7", "CO_ppm").show(10, truncate=False)

sampled_data = sampled_data.toPandas().set_index('Timestamp')
sampled_data = sampled_data[:int(len(sampled_data)/6)]
colors = sns.color_palette("bright", 5)
plt.figure(figsize=(15,8))
plt.plot(sampled_data['R7'], label='R7', color=colors[0], linewidth=1.5,alpha=0.8)
plt.plot(sampled_data['CO_ppm'], label='CO_ppm', color=colors[1], linewidth=1.5,alpha=0.8)
plt.title('Sampled data over 25 minutes')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid("x")
plt.tight_layout()
plt.savefig(os.path.join(plots_path, 'Sampled_data_over_25_minutes.png'), dpi=300)
plt.close()

spark.stop()