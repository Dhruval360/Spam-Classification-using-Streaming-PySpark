from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import StructType
import json

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

# Create a DStream that will connect to hostname:port
batch = ssc.socketTextStream("localhost", 6100)

dstream = batch.map(lambda x: json.loads(x[1]))
dstream.pprint()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate