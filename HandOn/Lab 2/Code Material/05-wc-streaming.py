"""
Structured Streaming Word‑Count Demo
===================================

Purpose
-------
This script demonstrates a minimal, end‑to‑end PySpark Structured Streaming
pipeline that counts words arriving on a local TCP socket in real time.
It is intentionally lightweight so you can run it from the command line
without any external infrastructure.

How to Run
----------
1. Start the socket source
    Open a second terminal and run:

    ```bash
    nc -lk 9999
    ```
    The nc command (Netcat) listens on TCP port 9999 and echoes any text you type.
    Whenever you hit Enter, a new line is sent to the Structured Streaming job.

2. Run the PySpark script
    The script will connect to the socket, split each incoming line into words,
    and maintain a running word‑frequency table.

3. Interact
    While the script is running, type any space‑separated words into the Netcat terminal
    (you can paste from sites such as https://www.lipsum.com). The console output will
    refresh every 2 seconds showing the current word counts. After 30 seconds the job
    will terminate automatically.
"""


from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("Structured Streaming Demo")
        .config("spark.executor.instances", "2")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

    # Read streaming data from a TCP socket
    lines = (
        spark.readStream.format("socket")
        .option("host", "localhost")
        .option("port", 9999)
        .load()
    )

    # Split each line into words, explode the array, then count
    words = lines.select(explode(split(col("value"), " ")).alias("word"))
    word_counts = words.groupBy("word").count()

    query = (
        word_counts.writeStream.outputMode("complete")
        .format("console")
        .option("truncate", "false")
        .trigger(processingTime="2 seconds")
        .start()
    )

    logger.debug("Streaming started – type words into the socket (nc -lk 9999).")
    query.awaitTermination(30)  # run for 30 seconds, then stop

    spark.stop()


if __name__ == "__main__":
    main()
