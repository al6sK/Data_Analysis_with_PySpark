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


# =======================================================
# FIX FOR WINDOWS ONLY                                  #
# =======================================================
os.environ['HADOOP_HOME'] = "C:\\hadoop"                #
sys.path.append("C:\\hadoop\\bin")                      #
os.environ['PATH'] += os.pathsep + "C:\\hadoop\\bin"    #
# =======================================================

plots_path = 'PLOTS/02_FEATURE ENGINEERING'
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

# =======================================
# FEATURE ENGINEERING
# =======================================
data = df_final.toPandas().set_index('Timestamp')
print(data.head())
print(data.shape)

# Creating 3 new contextual features.
# The first contextual feature is equal to the Temperature difference between consecutive time points.
data["Temperature_diff"] = data["Temperature"].shift(1) - data["Temperature"]
data.dropna(inplace = True)

# The second one is a binary option 0/1 characterizing the Heater_voltage to low or high.
data["Heater_voltage_state"] = [1 if v > 0.55 else 0 for v in data['Heater_voltage']]

# And the third one is equal to the mean of the 14 sensors
sensors = ["R" + str(i) for i in range(1,15)]
data["Sensors_Mean"] = data[sensors].mean(axis=1)

print(data[["Temperature_diff","Heater_voltage_state","Sensors_Mean"]].head(15))

# normalized values to 0-1
columns = ["CO_ppm","Sensors_Mean"]
norm_data = data[columns].copy()

for feature in columns:
    norm_data[feature] = (norm_data[feature] - norm_data[feature].min()) / (norm_data[feature].max() - norm_data[feature].min())

# plot CO_ppm_vs_Sensors_Mean
colors = sns.color_palette("bright", 2)

data = norm_data[:100]
plt.figure(figsize=(15,8))
plt.plot(data['CO_ppm'], label='CO_ppm', color=colors[0], linewidth=1.5,alpha=0.8)
plt.plot(data['Sensors_Mean'], label='Sensors_Mean', color=colors[1], linewidth=1.5,alpha=0.8)
plt.title('CO_ppm vs Sensors_Mean')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid("x")
plt.tight_layout()
plt.savefig(os.path.join(plots_path, 'CO_ppm_vs_Sensors_Mean5.png'), dpi=300)
plt.close()
