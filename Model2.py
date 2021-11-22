# importing required libraries
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.types import *
from pyspark.sql import functions as F

# from pyspark.ml import Pipeline
# from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
# from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
# from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row
import json

RED = '\033[91m'
RESET = '\033[0m'

# initializing spark session
sc = SparkContext(appName="PySparkShell")
spark = SparkSession(sc)
 
# my_schema = tp.StructType([
#   tp.StructField(name= 'Subject',  dataType= tp.StringType(),   nullable= True),
#   tp.StructField(name= 'Message',  dataType= tp.StringType(),   nullable= True),
#   tp.StructField(name= 'Spam/Ham', dataType= tp.StringType(),   nullable= True)
# ])
    
# define a function to compute sentiments of the received tweets
def get_prediction(data):
    try:
    # filter the tweets whose length is greater than 0
        # data = data.filter(lambda x: len(x) > 0)
    # create a dataframe with column name 'tweet' and each row will contain the tweet
        rowRdd = data.map(lambda w: Row(Subject = data['feature0'], ))
    # create a spark dataframe
        wordsDataFrame = spark.createDataFrame(rowRdd)
        wordsDataFrame.show()
    # transform the data using the pipeline and get the predicted sentiment
        # pipelineFit.transform(wordsDataFrame).select('tweet','prediction').show()
    except Exception as e: 
        print(RED, e, RESET)
    
# initialize the streaming context 
ssc = StreamingContext(sc, batchDuration= 3)

# Create a DStream that will connect to hostname:port, like localhost:9991
dstream = ssc.socketTextStream("localhost", 6100)

# split the tweet text by a keyword 'TWEET_APP' so that we can identify which set of words is from a single tweet
# batch = dstream.flatMap(lambda data : json.loads(data))

def readMyStream(rdd):
  if not rdd.isEmpty():
    df = spark.read.json(rdd, multiLine=True)

    df = df.withColumn("data", F.explode(F.arrays_zip("feature0", "feature1", "feature2")))\
           .select("data.feature0", "data.feature1", "data.feature2")


    print(RED)
    df.printSchema()
    print(RESET)
    print('Started the Process')
    print('Selection of Columns')
    # df = df.select('feature0', 'feature1', 'feature2')
    df.show(5)

dstream.foreachRDD( lambda rdd: readMyStream(rdd) )


# get the predicted sentiments for the tweets received
# batch.foreachRDD(get_prediction)

# Start the computation
ssc.start()             

# Wait for the computation to terminate
ssc.awaitTermination()  