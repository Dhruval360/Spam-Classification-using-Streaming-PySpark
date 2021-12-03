from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import col, when, explode, arrays_zip

import re
import joblib
import argparse
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import HashingVectorizer

# from pyspark.ml import Pipeline
# from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
# from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
# from pyspark.ml.classification import LogisticRegression

# Color coding the output to make it easier to find amongst the verbose output of Spark
RED = '\033[91m'
RESET = '\033[0m'
GREEN = '\033[92m'

hvec = HashingVectorizer(n_features = 2**9, alternate_sign = False)
stemmer = PorterStemmer()

models = {
    'kmeans' : MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6)
}


patterns = (
    re.compile(r'\\r\\n'),
    re.compile(r'[^a-zA-Z]'),
    re.compile(r'\s+'),
    re.compile(r'^b\s+'),
)

def proc(record):
    review = record.lower()
    for pattern in patterns: review = pattern.sub(' ', review)
    
    global stemmer
    review = [stemmer.stem(word) for word in review.split() if word not in stopwords.words('english')]
    review = ' '.join(review)
    return review

# proc = np.vectorize(proc)

from multiprocessing import Pool, cpu_count
pool = Pool(cpu_count()) 


def preProcess(X):
    # corpus = proc(X.reshape((len(X),)))
    corpus = list(pool.map(proc, X.reshape((len(X),))))    

    # Creating the Bag of Words model
    global hvec
    X = hvec.fit_transform(corpus)
    return X.toarray()

batchNum = 1
def trainBatch(rdd):
    if not rdd.isEmpty():
        df = (
            spark.read.json(rdd, multiLine = True)
            .withColumn("data", explode(arrays_zip("feature0", "feature1", "feature2")))
            .select("data.feature1", "data.feature2")
        )

        X = preProcess(np.array(df.select("feature1").collect()))
        y = np.array(
            df.withColumn("feature2", when(col("feature2") == 'spam', 1).otherwise(0))
                .select("feature2")
                .collect()
        )    

        global models, batchNum

        for model in models:
            models[model] = models[model].partial_fit(X)
            pred = models[model].predict(X)

            accuracy = accuracy_score(y, pred)
            precision = precision_score(y, pred, labels = np.unique(y))
            recall = recall_score(y, pred, labels = np.unique(y))
            conf_m = confusion_matrix(y, pred)

            print(GREEN)
            print(f"Model = {model}")
            print(f"accuracy: %.3f" %accuracy)
            print(f"precision: %.3f" %precision)
            print(f"recall: %.3f" %recall)
            print(f"confusion matrix: ")
            print(conf_m)
            print("\n\nSaving Model to disk...")

            joblib.dump(models[model], f"./TrainingLogs/{model}/Models/{batchNum}.sav")
            
            print("Model saved to disk")
            print(RESET)

            # with open(f"./TrainingLogs/{model}/logs.txt", "a") as f:
            #     f.write(f"Batch {batchNum}")
            #     f.write(f"\naccuracy: %.3f" %accuracy)
            #     f.write(f"\nprecision: %.3f" %precision)
            #     f.write(f"\nrecall: %.3f" %recall)
            #     f.write(f"\nconfusion matrix:\n")
            #     f.write(str(conf_m))
            #     f.write("\n-----------------------------------------------\n\n")
            
            with open(f"./TrainingLogs/{model}/logs.csv", "a") as f:
                f.write(f"{batchNum},{accuracy},{precision},{recall}\n")

        batchNum += 1

numBatches = None
def testBatch(rdd): # TODO
    if not rdd.isEmpty():
        df = (
            spark.read.json(rdd, multiLine = True)
            .withColumn("data", explode(arrays_zip("feature0", "feature1", "feature2")))
            .select("data.feature1", "data.feature2")
        )

        X = preProcess(df.feature1, df.count())
        y = np.array(
            df.withColumn("feature2", when(col("feature2") == 'spam', 1).otherwise(0))
            .select("feature2")
            .collect()
        )

        global numBatches
        for model in models:
            for batchNum in range(numBatches):
                curModel = joblib.load(f"./TrainingLogs/{model}/Models/{batchNum}.sav")
                pred = curModel.predict(X)

                accuracy = accuracy_score(y, pred)
                precision = precision_score(y, pred, pos_label = 1)
                recall = recall_score(y, pred, pos_label = 1)
                conf_m = confusion_matrix(y, pred)

                print(GREEN)
                print(f"Model = {model}")
                print(f"accuracy: %.3f" %accuracy)
                print(f"precision: %.3f" %precision)
                print(f"recall: %.3f" %recall)
                print(f"confusion matrix: ")
                print(conf_m)
                print(RESET)

                # with open(f"./TestingLogs/{model}/logs.txt", "a") as f:
                #     f.write(f"Batch {batchNum}")
                #     f.write(f"\naccuracy: %.3f" %accuracy)
                #     f.write(f"\nprecision: %.3f" %precision)
                #     f.write(f"\nrecall: %.3f" %recall)
                #     f.write(f"\nconfusion matrix:\n")
                #     f.write(str(conf_m))
                #     f.write("\n-----------------------------------------------\n\n")
                with open(f"./TestingLogs/{model}/logs.csv", "a") as f:
                    f.write(f"{batchNum},{accuracy},{precision},{recall}\n")

parser = argparse.ArgumentParser(description = 'Trains and Tests multiple models using PySpark')
parser.add_argument(
    '--delay', '-d',
    help = 'Delay in seconds before processing the next batch',
    required = False,
    type = int,
    default = 5
)
parser.add_argument(
    '--mode', '-m',
    help = 'Mode of operation, Training [train] or Testing [test]',
    required = False,
    type = str,
    default = 'train'
)
parser.add_argument(
    '--clean', '-c',
    help = 'Clear all logs and start a fresh training session',
    required = False,
    type = int,
    default = 0
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    batchDuration = args.delay
    mode = args.mode

    if(args.clean): # Clear all logs and start afresh
        for model in models:
            f = open(f"./TrainingLogs/{model}/logs.txt", "w")
            f.close()
            f = open(f"./TrainingLogs/{model}/logs.csv", "w")
            f.write("BatchNum,Accuracy,Precision,Recall\n")
            f.close()
        exit(0) # Will be removed later

    # Initializing the spark session
    sc = SparkContext(appName = "Spam Classifier")
    spark = SparkSession(sc)
    spark.sparkContext.setLogLevel('WARN')

    # Initializing the streaming context 
    ssc = StreamingContext(sc, batchDuration = batchDuration)

    # Create a DStream that will connect to hostname:port, like localhost:9991
    dstream = ssc.socketTextStream("localhost", 6100)
    
    if mode.lower() == 'train': dstream.foreachRDD(lambda rdd: trainBatch(rdd))
    elif mode.lower() == 'test': dstream.foreachRDD(lambda rdd: testBatch(rdd))
    else: raise("Invalid argument to the argument mode of operation: Use '-m train' or '-m test'")

    ssc.start()            # Start the computation
    ssc.awaitTermination() # Wait for the computation to terminate