from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import StructType
import json

from pyspark.sql import SparkSession
  
# creating sparksession and giving an app name
# spark = SparkSession.builder.appName('sparkdf').getOrCreate()


# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "Spam Classifier")
ssc = StreamingContext(sc, 5)
sc.setLogLevel("WARN") # To reduce the verbosity of spark-submit

# Create a DStream that will connect to hostname:port
dstream = ssc.socketTextStream("localhost", 6100)


ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate