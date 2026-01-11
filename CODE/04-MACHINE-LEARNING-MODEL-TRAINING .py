from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, lag
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
    .csv("DATA/data_Downsampled.csv") \

print(data.columns)

# ===========================================================================================
# Data preparacion
# ===========================================================================================
# melt data
r_columns = ["R" + str(i) for i in range(1, 15)]
data = data.unpivot(
    ids=['Timestamp', 'CO_ppm', 'Humidity', 'Temperature', 'Flow_rate', 
         'Heater_voltage', 'Temperature_diff', 'Heater_voltage_state', 'Sensors_Mean'], 
    values=r_columns, 
    variableColumnName="Sensor_ID", 
    valueColumnName="R"
)

# save lag values 
windowSpec = Window.partitionBy("Sensor_ID").orderBy("Timestamp")
data = data.withColumn("Prev_Sensors_Mean", lag("Sensors_Mean", 1).over(windowSpec))\
            .withColumn("Prev_R", lag("R", 1).over(windowSpec)) 

data = data.drop("Sensors_Mean").na.drop()

data.select("Timestamp", "Sensor_ID", "R", "Prev_Sensors_Mean").show(5)

# assembler
input_cols = ['CO_ppm', 'Humidity', 'Temperature', 'Flow_rate','Heater_voltage',
              'Temperature_diff', 'Heater_voltage_state', 'Prev_Sensors_Mean','Prev_R']
assembler = VectorAssembler(inputCols = input_cols, outputCol = "features_raw")

# Normalization
scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")


# Data spliting
w = Window.orderBy("Timestamp")
data = data.withColumn("rank", F.percent_rank().over(w))

train = data.filter(F.col("rank") <= 0.9).cache()
# val = data.filter((F.col("rank") > 0.7) & (F.col("rank") <= 0.9)).cache()
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
    .addGrid(rf.numTrees, [25,50])           
    .addGrid(rf.maxDepth, [10,15])            
    .build()
)

results = []

for param_grid, model in zip([lr_param_grid,rf_param_grid],[lr,rf]):
    model_name = model.__class__.__name__
    # Build the pipeline
    pipeline = Pipeline(stages=[assembler, scaler, model])

    # Set up the CrossValidator for robustness
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator_rmse,  # BinaryClassificationEvaluator
        numFolds=3,
    )
    print(f"Running cross‑validation for {model_name}…")
    cv_model = cv.fit(train)
    best_predictions = cv_model.transform(test)

    rmse = evaluator_rmse.evaluate(best_predictions)
    mae  = evaluator_mae.evaluate(best_predictions)
    r2   = evaluator_r2.evaluate(best_predictions)

    best_model_stage = cv_model.bestModel.stages[-1]
    results.append((model_name, rmse, mae, r2, best_model_stage))

print("\n" + "="*50)
print(f"{'Model':<25} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10}")
print("-" * 65)

for name, rmse, mae, r2, model in results:
    print(f"{name:<25} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f}")
    
    if "RandomForest" in name:
        trees = model.getOrDefault("numTrees")
        depth = model.getOrDefault("maxDepth")
        print(f"   --> Best Params: Trees={trees}, Depth={depth}")
        
    elif "LinearRegression" in name:
        reg = model.getOrDefault("regParam")
        elastic = model.getOrDefault("elasticNetParam")
        print(f"   --> Best Params: RegParam={reg}, ElasticNet={elastic}")
print("="*50)

spark.stop()