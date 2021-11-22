from pyspark import SparkConf,SparkContext  #single application
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType #for every user
from pyspark.streaming import StreamingContext #for streaming data
import json
import pickle

sc = SparkContext("local[2]","spam_ham") #paralleyly run spark on two threads
spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate() #create an spark session for the user if already not existsing
spark.sparkContext.setLogLevel('WARN')

ssc = StreamingContext(sc,7) #streaming context ie entry point for fstreaming functionalities

tableSchema = StructType([
    StructField('subject',StringType(),True),
    StructField("message",StringType(),True),
    StructField("spam/ham",StringType(),True)
])

df = None

def preprocess(row):
    row_list = json.loads(row)
    with open("logging.txt","w") as f:
        f.write(str(row_list))
        f.write("\n")

    data = []
    for row in row_list:
        vals_list = row.split(",")
        for i in range(0,len(vals_list)):
            word = vals_list[i]
            vals_list[i] = word.strip('\n')
        data.append(tuple(vals_list))
    with open("logging.txt","a") as f:
        f.write(str(data))
        f.write("\n")

        global df
        df = spark.createDataFrame(data=data,schema=tableSchema)
        with open("logging.txt","a") as f:
            f.write(str(df))
            f.write("\n")
            df.printSchema()
    return row_list

if __name__=="__main__":
    txt_lines = ssc.socketTextStream('localhost',6100)
    csv_rows = txt_lines.flatMap(lambda line : line.split("\n")).map(lambda x: preprocess(x))
    csv_rows.pprint()

    ssc.start()
    ssc.awaitTermination()

    



