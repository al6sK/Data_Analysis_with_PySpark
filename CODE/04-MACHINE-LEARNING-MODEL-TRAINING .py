from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, lag,expr
import multiprocessing
import os
import sys
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression,RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import StringIndexer

# =======================================================
# FIX FOR WINDOWS ONLY                                  #
# =======================================================
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'              #
os.environ['SPARK_DRIVER_HOST'] = '127.0.0.1'           #
os.environ['HADOOP_HOME'] = "C:\\hadoop"                #
sys.path.append("C:\\hadoop\\bin")                      #
os.environ['PATH'] += os.pathsep + "C:\\hadoop\\bin"    #
# =======================================================

plots_path = 'PLOTS/04_MACHINE_LEARNING_MODEL_TRAINING'
os.makedirs(plots_path, exist_ok=True)
sns.set_theme(style="whitegrid")

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
data = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("DATA/log_data_Downsampled.csv") \

print(data.columns)

# ===========================================================================================
# Data preparacion
# ===========================================================================================
# melt data
features = [
    "Timestamp","CO_ppm","Humidity","Temperature","Flow_rate","Heater_voltage",
    "Temperature_diff","Heater_voltage_state","Sensors_Mean","R1to7_Mean","R8to14_Mean",
    "R1to7_Mean_short_zscore","R1to7_Mean_medium_zscore","R1to7_Mean_long_zscore","R8to14_Mean_short_zscore",
    "R8to14_Mean_medium_zscore","R8to14_Mean_long_zscore"
]

stack_string = "stack(14, " + \
               ", ".join([f"'{i}', R{i}, Prev_R{i}" for i in range(1, 15)]) + \
               ") as (Sensor_ID, R, Prev_R)"

data = data.select(
    *features,    
    expr(stack_string)
)

# assembler
input_cols = [   
    'Prev_R',                   
    'CO_ppm', 'Humidity', 'Temperature', 'Flow_rate', 
    'Heater_voltage', 'Temperature_diff', 'Heater_voltage_state',
    # 'Sensors_Mean',
    'R1to7_Mean', 'R8to14_Mean',
    'R1to7_Mean_short_zscore', 'R1to7_Mean_medium_zscore', 'R1to7_Mean_long_zscore',
    'R8to14_Mean_short_zscore', 'R8to14_Mean_medium_zscore', 'R8to14_Mean_long_zscore'
]
assembler = VectorAssembler(inputCols = input_cols, outputCol = "features_raw")

# Normalization
scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")

# Data spliting
w = Window.orderBy("Timestamp")
data = data.withColumn("rank", F.percent_rank().over(w))

train = data.filter(F.col("rank") <= 0.7).cache()
val = data.filter((F.col("rank") > 0.7) & (F.col("rank") <= 0.9)).cache()
test = data.filter(F.col("rank") > 0.9).cache()

# print(f"Train rows: {train.count()}, Val rows: {val.count()}, Test rows: {test.count()}")

# ===========================================================================================
# Linear Regression and RandomForestRegressor
# ===========================================================================================
lr = LinearRegression(featuresCol="features", labelCol="R")
rf = RandomForestRegressor(featuresCol="features", labelCol="R", seed=42)

# Define the parameter grids
lr_param_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1])        
    .addGrid(lr.elasticNetParam, [0.0, 1.0]) 
    .build()
)

rf_param_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [200]) # 200         
    .addGrid(rf.maxDepth, [20]) # 20
    .build()
)
# ===========================================================================================
# Plot results
# ===========================================================================================
def plot_predictions_vs_real(predictions, model_name):
    data = predictions.select("Timestamp", "Sensor_ID", "R_real", "prediction_real").toPandas()
    data.sort_values(by=['Sensor_ID', 'Timestamp'], inplace=True)

    fig , axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 26), sharex=True)
    axes = axes.flatten()

    sensor_ids = sorted(data['Sensor_ID'].unique(), key=int)

    for i, sensor_id in enumerate(sensor_ids):
        ax = axes[i]
        sensor_data = data[data['Sensor_ID'] == sensor_id]
        l1, = ax.plot(sensor_data['Timestamp'], sensor_data['R_real'], label='Actual', color="black", alpha=0.7, linewidth=1)
        l2, = ax.plot(sensor_data['Timestamp'], sensor_data['prediction_real'],label='Predicted', color="red", alpha=0.7, linewidth=1, linestyle='--')
        ax.set_title(f"Sensor R{sensor_id}", fontsize=11, fontweight='bold', loc='left', color='black')
        if i == 0:
            handles = [l1, l2]
            labels  = [l1.get_label(), l2.get_label()]
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.legend(handles, labels, 
                loc='lower center',           
                bbox_to_anchor=(0.5, 0.93),   
                ncol=2, 
                frameon=True, 
                fontsize=13,
                facecolor='white',
                edgecolor='lightgray',
                borderpad=0.6)
        
    plt.suptitle(f"Model Accuracy Analysis: {model_name}", fontsize=25, fontweight='bold', y=0.98, color='#2C3E50')
    plt.savefig(os.path.join(plots_path, f"{model_name}_Performance_per_Sensor.png"), dpi=300)
    plt.close()
# ===========================================================================================
# Evaluation
# ===========================================================================================
evaluator_rmse = RegressionEvaluator(labelCol="R", predictionCol="prediction", metricName="rmse")

results = []
for param_grid, model in zip([lr_param_grid,rf_param_grid],[lr,rf]):
    model_name = model.__class__.__name__

    best_model = None
    best_rmse_val = float('inf')
    best_params = None

    for params in param_grid:
        temp_model = model.copy(params)
        # Build the pipeline
        pipeline = Pipeline(stages=[assembler, scaler, temp_model])
        # Fit train data to model
        fitted_model = pipeline.fit(train)
        # Predicte val data
        val_predictions = fitted_model.transform(val)
        rmse_val = evaluator_rmse.evaluate(val_predictions)

        if rmse_val < best_rmse_val:
            best_rmse_val = rmse_val
            best_model = fitted_model 
            best_params = params

    test_predictions = best_model.transform(test)
    # Reverse log real values and predicted
    test_predictions = test_predictions.withColumn("R_real", F.expm1("R"))\
                                        .withColumn("prediction_real", F.expm1("prediction"))
    # Evaluation
    real_evaluator_rmse = RegressionEvaluator(labelCol="R_real", predictionCol="prediction_real", metricName="rmse")
    real_evaluator_mae  = RegressionEvaluator(labelCol="R_real", predictionCol="prediction_real", metricName="mae")
    real_evaluator_r2   = RegressionEvaluator(labelCol="R_real", predictionCol="prediction_real", metricName="r2")

    test_mape = test_predictions.select(F.mean(F.abs((F.col("R_real") - F.col("prediction_real")) / F.col("R_real")))).collect()[0][0] * 100
    test_rmse = real_evaluator_rmse.evaluate(test_predictions)
    test_mae  = real_evaluator_mae.evaluate(test_predictions)
    test_r2   = real_evaluator_r2.evaluate(test_predictions)
    # Save results
    results.append((model_name, test_mape, test_rmse, test_mae, test_r2, best_params))

    # Plot Real VS Predictions values
    plot_predictions_vs_real(test_predictions, model_name)

print("\n" + "="*80)
print(f"{'Model':<20} | {'MAPE':<10} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10} | {'Best Params'}")
print("-" * 80)
for name, mape, rmse, mae, r2, params in results:
    param_str = str({p.name: v for p, v in params.items()})
    print(f"{name:<20} | {mape:<10.4f} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f} | {param_str}")
print("="*80)
spark.stop()