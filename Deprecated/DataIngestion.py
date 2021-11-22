from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import StructType
import json

from pyspark.sql import SparkSession

# creating sparksession and giving an app name
spark = SparkSession.builder.appName('sparkdf').getOrCreate()


# Create a local StreamingContext with two working thread and batch interval of 1 second
# sc = SparkContext("local[2]", "NetworkWordCount")
# ssc = StreamingContext(sc, 1)
# sc.setLogLevel("WARN") # To reduce the verbosity of spark-submit

# Create a DStream that will connect to hostname:port
batch = spark.readStream\
      .format("socket")\
      .option("host","localhost")\
      .option("port","6100")\
      .load()

query = batch\
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()# dstream = batch.flatMap(lambda x: json.loads(x))
# dstream.pprint()


# ssc.start()             # Start the computation
# ssc.awaitTermination()  # Wait for the computation to terminate
