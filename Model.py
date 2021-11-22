from pyspark import SparkConf,SparkContext  #single application
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType #for every user
from pyspark.streaming import StreamingContext #for streaming data
import json

sc = SparkContext("local[2]","spam_ham") #paralleyly run spark on two threads
spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate() #create an spark session for the user if already not existsing
spark.sparkContext.setLogLevel('WARN')

ssc = StreamingContext(sc,7) #streaming context ie entry point for fstreaming functionalities

tableSchema = StructType().add("subject","string").add("message","string").add("spam/ham","string")

if __name__=="__main__":
    txt_lines = ssc.socketTextStream('localhost',6100)
    csv_rows = txt_lines.flatMap(lambda line : line.split("\n")).map(lambda x: json.loads(x))
    csv_rows.pprint()

    ssc.start()
    ssc.awaitTermination()

    



