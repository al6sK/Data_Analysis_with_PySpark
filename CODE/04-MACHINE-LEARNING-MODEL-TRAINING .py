from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, lag, expr, col, abs, pow, sqrt, mean, stddev
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
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.feature import StringIndexer
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score

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
    .getOrCreate()
)

sc = spark.sparkContext
data = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("DATA/log_data_Downsampled.csv")

# ===========================================================================================
# Data preparation
# ===========================================================================================
# melt data
features = [
    "Timestamp","CO_ppm","Humidity","Temperature","Flow_rate","Heater_voltage",
    "Temperature_diff","Heater_voltage_state","Sensors_Mean","R1to7_Mean","R8to14_Mean",
    "R1to7_Mean_short_zscore","R1to7_Mean_medium_zscore","R1to7_Mean_long_zscore","R8to14_Mean_short_zscore",
    "R8to14_Mean_medium_zscore","R8to14_Mean_long_zscore"
]

stack_string = "stack(14, " + \
               ", ".join([f"'{i}', Target_R{i}, R{i}" for i in range(1, 15)]) + \
               ") as (Sensor_ID, Target_R, R)"

data = data.select(
    *features,    
    expr(stack_string)
)

indexer = StringIndexer(inputCol="Sensor_ID", outputCol="Sensor_ID_Index")
data = indexer.fit(data).transform(data)

# assembler
input_cols = [   
    'R', 
    'Sensor_ID_Index',                  
    'CO_ppm', 'Humidity', 'Temperature', 'Flow_rate', 
    'Heater_voltage', 'Temperature_diff', 'Heater_voltage_state',
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

# ===========================================================================================
# Linear Regression and RandomForestRegressor
# ===========================================================================================
lr = LinearRegression(featuresCol="features", labelCol="Target_R")
rf = RandomForestRegressor(featuresCol="features", labelCol="Target_R", seed=42)

# Define the parameter grids
lr_param_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1])        
    .addGrid(lr.elasticNetParam, [0.0, 1.0]) 
    .build()
)

rf_param_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [200])         
    .addGrid(rf.maxDepth, [20]) # 20
    .build()
)

# ===========================================================================================
# Plot results Function
# ===========================================================================================
def plot_predictions_vs_real(predictions, model_name):
    data_pd = predictions.select("Timestamp", "Sensor_ID", "R_real", "prediction_real").toPandas()
    data_pd['Sensor_ID'] = data_pd['Sensor_ID'].astype(int) # Ensure ID is int for sorting
    data_pd.sort_values(by=['Sensor_ID', 'Timestamp'], inplace=True)

    fig , axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 26), sharex=True)
    axes = axes.flatten()

    sensor_ids = sorted(data_pd['Sensor_ID'].unique())

    for i, sensor_id in enumerate(sensor_ids):
        ax = axes[i]
        sensor_data = data_pd[data_pd['Sensor_ID'] == sensor_id]
        
        l1, = ax.plot(sensor_data['Timestamp'], sensor_data['R_real'], label='Actual', color="black", alpha=0.7, linewidth=1)
        l2, = ax.plot(sensor_data['Timestamp'], sensor_data['prediction_real'],label='Predicted', color="red", alpha=0.7, linewidth=1, linestyle='--')
        
        ax.set_title(f"Sensor R{sensor_id}", fontsize=11, fontweight='bold', loc='left', color='black')
        
        if i == 0:
            handles = [l1, l2]
            labels  = [l1.get_label(), l2.get_label()]
            
    if len(sensor_ids) < len(axes):
        fig.delaxes(axes[-1])

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
# Main Loop for Training and Evaluation per Sensor
# ===========================================================================================
evaluator_rmse = RegressionEvaluator(labelCol="Target_R", predictionCol="prediction", metricName="rmse")

for param_grid, model in zip([lr_param_grid, rf_param_grid], [lr, rf]):
    model_name = model.__class__.__name__
    print(f"\nProcessing Model: {model_name}...")
    
    start_train = time.time()
    
    best_model = None
    best_rmse_val = float('inf')
    best_params = None

    # Cross-Validation manually (Train/Val split)
    for params in param_grid:
        temp_model = model.copy(params)
        pipeline = Pipeline(stages=[assembler, scaler, temp_model])
        fitted_model = pipeline.fit(train)
        val_predictions = fitted_model.transform(val)
        rmse_val = evaluator_rmse.evaluate(val_predictions)

        if rmse_val < best_rmse_val:
            best_rmse_val = rmse_val
            best_model = fitted_model 
            best_params = params

    end_train = time.time()
    train_time = end_train - start_train
    
    print("-" * 80)
    print(f"BEST PARAMS FOUND FOR {model_name}:")
    for p, v in best_params.items():
        print(f"  * {p.name}: {v}")
    print(f"  * Best Validation RMSE: {best_rmse_val:.4f}")
    print("-" * 80)

    # Predict on Test Data
    start_inf = time.time()
    test_predictions = best_model.transform(test)
    
    # Force execution to measure inference time
    test_count = test_predictions.count() 
    end_inf = time.time()
    inference_time = end_inf - start_inf
    
    # Reverse log transform
    test_predictions = test_predictions.withColumn("R_real", F.expm1("Target_R"))\
                                       .withColumn("prediction_real", F.expm1("prediction"))

    # =======================================================================================
    # Calculate Metrics PER SENSOR 
    # =======================================================================================
    # Calculate Error = True - Pred
    test_predictions = test_predictions.withColumn("error", col("R_real") - col("prediction_real"))
    
    # Calculate per sensor metrics
    metrics_per_sensor = test_predictions.groupBy("Sensor_ID").agg(
        F.mean(F.abs(col("error"))).alias("MAE"),
        (F.mean(F.abs(col("error") / col("R_real"))) * 100).alias("MAPE"),
        F.sqrt(F.mean(F.pow(col("error"), 2))).alias("RMSE"),
        F.mean("error").alias("Mean_Error"),
        F.stddev("error").alias("Std_Error")
    )
    
    preds_pd = test_predictions.select("Sensor_ID", "R_real", "prediction_real", "error").toPandas()
    
    final_metrics = []
    
    for sid in sorted(preds_pd['Sensor_ID'].unique(), key=int):
        df_s = preds_pd[preds_pd['Sensor_ID'] == sid]
        y_true = df_s['R_real']
        y_pred = df_s['prediction_real']
        errors = df_s['error']
        
        # Recalculate metrics in Pandas to be sure and get R2
        mae = round(mean_absolute_error(y_true, y_pred), 4)
        rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
        mape = round(mean_absolute_percentage_error(y_true, y_pred) * 100, 4)
        r2 = round(r2_score(y_true, y_pred), 4)
        mean_err = round(float(np.mean(errors)), 4)
        std_err = round(float(np.std(errors)), 4)

        final_metrics.append({
            "Sensor_ID": f"Sensor_R{sid}",
            "MAPE (%)": mape,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Mean_Error": mean_err,
            "Std_Error": std_err
        })
        
    metrics_df = pd.DataFrame(final_metrics)
    metrics_df.set_index("Sensor_ID", inplace=True)

    print("\n" + "="*80)
    print(f" {model_name} PER SENSOR RESULTS")
    print(f" Training Time: {train_time:.2f} s | Inference Time: {inference_time:.4f} s")
    print("="*80)
    print(metrics_df)
    print("="*80)
    
    # Save metrics to CSV
    metrics_df.to_csv(
    os.path.join(plots_path, f"{model_name}_Metrics_Per_Sensor.csv"), 
    sep=';',     
    decimal=','    
    )
    # Plot Real VS Predictions values
    plot_predictions_vs_real(test_predictions, model_name)

spark.stop()