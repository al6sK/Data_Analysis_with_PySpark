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

plots_path = 'PLOTS/04-MACHINE-LEARNING-MODEL-TRAINING '
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
    'Sensors_Mean',
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

evaluator_rmse = RegressionEvaluator(labelCol="R", predictionCol="prediction", metricName="rmse")
evaluator_mae  = RegressionEvaluator(labelCol="R", predictionCol="prediction", metricName="mae")
evaluator_r2   = RegressionEvaluator(labelCol="R", predictionCol="prediction", metricName="r2")

# Define the parameter grids
lr_param_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1])        
    .addGrid(lr.elasticNetParam, [0.0, 1.0]) 
    .build()
)

rf_param_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [1000]) # 50 ,200         
    .addGrid(rf.maxDepth, [17]) # 17 
    .build()
)

# ===========================================================================================
# Evaluation
# ===========================================================================================
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
    
    test_mape = test_predictions.select(
        F.mean(F.abs((F.col("R") - F.col("prediction")) / F.col("R")))
    ).collect()[0][0] * 100
    test_rmse = evaluator_rmse.evaluate(test_predictions)
    test_mae  = evaluator_mae.evaluate(test_predictions)
    test_r2   = evaluator_r2.evaluate(test_predictions)
    
    results.append((model_name,test_mape,  test_rmse, test_mae, test_r2, best_params))

print("\n" + "="*80)
print(f"{'Model':<20} | {'MAPE':<10} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10} | {'Best Params'}")
print("-" * 80)
for name, mape, rmse, mae, r2, params in results:
    param_str = str({p.name: v for p, v in params.items()})
    print(f"{name:<20} | {mape:<10.4f} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f} | {param_str}")
print("="*80)

spark.stop()