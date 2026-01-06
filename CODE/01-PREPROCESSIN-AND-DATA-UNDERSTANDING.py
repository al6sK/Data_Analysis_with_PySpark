from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
import multiprocessing

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

    .config("spark.ui.showConsoleProgress", "true")
    .getOrCreate()
)

sc = spark.sparkContext

df_raw = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("DATA/downloads/*.csv") \
    .withColumn("source_file", input_file_name())

print(f"Συνολικές εγγραφές: {df_raw.count()}")
df_raw.printSchema()












spark.stop()