from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import col, when, explode, arrays_zip, concat, lit

import re
import shutil
import joblib
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore') # To ignore the UndefinedMetricWarning

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
engStopWords = stopwords.words('english') # List of english stop words that would be used during preprocessing

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import MiniBatchKMeans

# Color coding the output to make it easier to find amongst the verbose output of Spark
ORANGE = '\033[33m'
RESET = '\033[0m'
GREEN = '\033[92m'

# Tokens are encoded as numerical indices using hash function but once hashed, multiple tokens can map to same index so they cannot be retrieved.
# Number of feature columns used is 512.
hvec = HashingVectorizer(n_features = 2**9, alternate_sign = False)
stemmer = PorterStemmer() # Stems the words. Eg: Converts running, ran, run to run.

'''
Multinomial Naive Bayes: Bayesian classifier for discrete features.
Perceptron: Linear perceptron classifier
Multi layer Perceptron: Multi layer perceptron classifier
'''
classifiers = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'SGD Classifier': SGDClassifier(
                            loss = 'log', 
                            alpha = 0.0001, 
                            max_iter = 3000, 
                            tol = None, 
                            shuffle = True, 
                            verbose = 0, 
                            learning_rate = 'adaptive', 
                            eta0 = 0.01, 
                            early_stopping = False
                      ),
    'Perceptron': Perceptron(),
    'Multi Layer Perceptron': MLPClassifier(
                                activation = 'tanh', 
                                learning_rate = 'adaptive',
                                alpha = 1e-4, 
                                hidden_layer_sizes = (15,), 
                                random_state = 1, 
                                verbose = False,
                                max_iter = 1, 
                                warm_start = True
                              )
}

clustering_model = MiniBatchKMeans(n_clusters=2, random_state=0)

patterns = ( 
    re.compile(r'\\r\\n'),    # Select carriage returns and new lines
    re.compile(r'[^a-zA-Z]'), # Select anything that isn't an alphabet
    re.compile(r'\s+'),       # Select multiple consecutive spaces
)

def preProcess(record):
    '''
    Performs Text Preprocessing
    * Substitutes carriage returns, new lines, non alphabetic characters and multiple consecutive spaces with a single space ' '.
    * Stems the words in the cleaned text
    Returns the preprocessed string
    '''
    text = record.lower()
    for pattern in patterns: text = pattern.sub(' ', text)

    global stemmer
    return ' '.join([stemmer.stem(word) for word in text.split() if word not in engStopWords])

def readStream(rdd):
    '''
    Reads a JSON RDD into a DataFrame, performs preprocessing
    Returns Features X and Labels y
    '''
    global hvec
    '''
    Array_zip combines the ith position elements of all three features into a list.
    Hence there would be a list of such lists at the end of Array_Zip.
    Explode converts that into three columns with rows as values
    '''
    df = (
            spark.read.json(rdd, multiLine = True)
            .withColumn("data", explode(arrays_zip("feature0", "feature1", "feature2")))
            .select("data.feature0", "data.feature1", "data.feature2")
        )

    df = df.withColumn('joint', concat(col('feature0'), lit(" "), col('feature1'))) # Concatenating the Subject and the Message
    X = hvec.fit_transform( # Fitting the HashVectorizer on the preprocessed strings
            np.array(
                df.select("joint").rdd.map(lambda x : preProcess(str(x))).collect() # Preprocessing each Record parallely
            )
        ).toarray()
    y = np.array(
        df.withColumn("feature2", when(col("feature2") == 'spam', 1).otherwise(0))  # Encoding spam and ham as 1 and 0 respectively
            .select("feature2")
            .collect()
    )    
    return X, y

batchNum = 1
def trainBatch(rdd):
    '''
    Trains Multiple models on the given batch of data.
    Saves the trained models and logs the training metrics obtained.
    '''
    if not rdd.isEmpty():
        global classifiers, batchNum
        X, y = readStream(rdd)
        print(ORANGE)
        print(f"Processing Batch {batchNum} of size {len(X)}\n")



        for model in classifiers:
            print(GREEN)
            classifiers[model] = classifiers[model].partial_fit(X, y.reshape((len(y),)), np.unique(y))
            pred = classifiers[model].predict(X)

            accuracy = accuracy_score(y, pred)
            precision = precision_score(y, pred, labels = np.unique(y))
            recall = recall_score(y, pred, labels = np.unique(y))
            conf_m = confusion_matrix(y, pred)

            print(f"Model = {model}")
            print(f"accuracy: %.3f" %accuracy)
            print(f"precision: %.3f" %precision)
            print(f"recall: %.3f" %recall)
            print(f"confusion matrix: ")
            print(conf_m)
            print("\n\nSaving Model to disk... ", end = '')

            joblib.dump(classifiers[model], f"./Logs/{model}/Models/{batchNum}.sav") # Saving the trained model
            shutil.copyfile(f"./Logs/{model}/Models/{batchNum}.sav", f"./Logs/{model}/final_model.sav")
            
            print("Model saved to disk!")
            print(RESET)

            # with open(f"./Logs/{model}/TrainLogs/logs.txt", "a") as f:
            #     f.write(f"Batch {batchNum}")
            #     f.write(f"\naccuracy: %.3f" %accuracy)
            #     f.write(f"\nprecision: %.3f" %precision)
            #     f.write(f"\nrecall: %.3f" %recall)
            #     f.write(f"\nconfusion matrix:\n")
            #     f.write(str(conf_m))
            #     f.write("\n-----------------------------------------------\n\n")
            
            with open(f"./Logs/{model}/TrainLogs/logs.csv", "a") as f:
                f.write(f"{batchNum},{accuracy},{precision},{recall}\n")
        
        batchNum += 1

numBatches = None

batchNum = 1
def trainCluster(rdd):
    if not rdd.isEmpty():
        global clustering_model, batchNum
        X, y = readStream(rdd)
        print(ORANGE)
        print(f"Processing Batch {batchNum} of size {len(X)}\n")


        clustering_model = clustering_model.partial_fit(X)
        pred = clustering_model.predict(X)

        accuracy_spam_1 = accuracy_score(y, pred)
        precision_spam_1 = precision_score(y, pred, labels = np.unique(y))
        recall_spam_1 = recall_score(y, pred, labels = np.unique(y))
        conf_m_spam_1 = confusion_matrix(y, pred)
        
        print(GREEN)
        print(f"Model = Clustering")
        print(f"MEASURES WHEN SPAM IS ENCODED AS 1")
        print(f"accuracy: %.3f" %accuracy_spam_1)
        print(f"precision: %.3f" %precision_spam_1)
        print(f"recall: %.3f" %recall_spam_1)
        print(f"confusion matrix: ")
        print(conf_m_spam_1)
        
        print(ORANGE)
        for i in y:
            if(i[0] == 1):
                i[0] = 0
            else:
                i[0] = 1
        

        accuracy_spam_0 = accuracy_score(y, pred)
        precision_spam_0 = precision_score(y, pred, labels = np.unique(y))
        recall_spam_0 = recall_score(y, pred, labels = np.unique(y))
        conf_m_spam_0 = confusion_matrix(y, pred)

        print(f"MEASURES WHEN SPAM IS ENCODED AS 0")
        print(f"accuracy: %.3f" %accuracy_spam_0)
        print(f"precision: %.3f" %precision_spam_0)
        print(f"recall: %.3f" %recall_spam_0)
        print(f"confusion matrix: ")
        print(conf_m_spam_0)
        
        print("\n\nSaving Model to disk...")

        joblib.dump(clustering_model, f"./Logs/Clustering/Models/{batchNum}.sav")
        shutil.copyfile(f"./Logs/Clustering/Models/{batchNum}.sav", f"./Logs/Clustering/final_model.sav")
        print("Model saved to disk")
        print(RESET)
        
        with open(f"./Logs/Clustering/TrainLogs/logs.csv", "a") as f:
            f.write(f"{batchNum},{accuracy_spam_1},{precision_spam_1},{recall_spam_1},{accuracy_spam_0},{precision_spam_0},{recall_spam_0}\n")

        batchNum += 1

numBatches = None


batchNum = 1
def testBatch(rdd, cluster = 0, clusterlogfile = None, logfiles = None):
    '''
    Input:
        rdd: The test rdd upon which the model to run
        cluster: Whether to use clustering / other models
        clusterlogfile: File to which the clustering model's metrics are logged to
        logfiles: A dictionary of log files pertaining to each supervised model present here.

    Output:
        Logs the information [batch number, prediction value, actual value] into the csv file in TrainLogs for each classifier
    '''
    if not rdd.isEmpty():
        global batchNum, classifiers, clustering_model

        # Read stream and pre-process it
        X, gt_values = readStream(rdd)

        print(ORANGE)
        print(f"Processing Batch {batchNum} of size {len(X)}\n")
        
        print(GREEN)
        if(cluster == 0):
            # For all the classifiers load the right model, predict on the test rdd (i.e. X), Log to file
            for model in classifiers:                
                predictions = classifiers[model].predict(X)
                print(f"Model = {model}")
                for i in zip(predictions, gt_values):
                    logfiles[model].write(f"{batchNum},{i[0]},{i[1][0]}\n")
                    logfiles[model].flush()
                print(f"Successfully logged to file...\n")
                
        else:            
            predictions = clustering_model.predict(X)
            print(f"Model = Clustering")
            for i in zip(predictions, gt_values):
                clusterlogfile.write(f"{batchNum},{i[0]},{i[1][0]}\n")
                clusterlogfile.flush()
            print(f"Successfully logged to file...\n")
        
        print(RESET)
        batchNum += 1

parser = argparse.ArgumentParser(description = 'Trains and Tests multiple classifiers using PySpark')
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
    help = 'Clear all logs',
    required = False,
    type = int,
    default = 0
)

parser.add_argument(
    '--clustering', '-t',
    help = 'Train a k means cluster model, default = 0',
    required = False,
    type = int,
    default = 0
)

parser.add_argument(
    '--model_num', '-mn',
    help = 'Use models from this batch for testing',
    required = False,
    type = int,
    default = 0
)



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    batchDuration = args.delay
    mode = args.mode
    cluster = args.clustering

    if(args.clean): # Clear all logs
        for model in classifiers:
            with open(f"./Logs/{model}/TrainLogs/logs.csv", "w") as f: f.write("BatchNum,Accuracy,Precision,Recall\n")
            with open(f"./Logs/{model}/TestLogs/logs.csv", "w") as f: f.write("BatchNum,Prediction,GroundTruth\n")
            
        with open(f"./Logs/Clustering/TrainLogs/logs.csv", "w") as f: f.write("batchNum,accuracy_spam_1,precision_spam_1,recall_spam_1,accuracy_spam_0,precision_spam_0,recall_spam_0\n")
        with open(f"./Logs/Clustering/TestLogs/logs.csv", "w") as f: f.write("BatchNum,Prediction,GroundTruth\n")

        print(f"\n{GREEN}Cleared the Logs{RESET}\n")
        exit(0)

    # Initializing the spark session
    sc = SparkContext(appName = "Spam Classifier")
    spark = SparkSession(sc)
    spark.sparkContext.setLogLevel('WARN')

    # Initializing the streaming context 
    ssc = StreamingContext(sc, batchDuration = batchDuration)

    # Create a DStream that will connect to hostname:port
    dstream = ssc.socketTextStream("localhost", 6100)
    
    if mode.lower() == 'train': 
        if(cluster):
            dstream.foreachRDD(lambda rdd: trainCluster(rdd))
        else:
            dstream.foreachRDD(lambda rdd: trainBatch(rdd))
    
    elif mode.lower() == 'test': 
        model_num = int(args.model_num)
        clusterlogfile = None
        logfiles = None
        if cluster:
            if(model_num == 0):
                clusterlogfile = open(f"./Logs/Clustering/TestLogs/logs.csv", "w")
                clusterlogfile.write("BatchNum,Prediction,GroundTruth\n") # Clearing logs
                clustering_model = joblib.load(f"./Logs/Clustering/final_model.sav")
            else:
                clusterlogfile = open(f"./Logs/Clustering/TestLogs/logs{model_num}.csv", "w")
                clusterlogfile.write("BatchNum,Prediction,GroundTruth\n") # Clearing logs
                clustering_model = joblib.load(f"./Logs/Clustering/Models/{model_num}.sav")

            clusterlogfile.flush()
        else:
            logfiles = {}
            if model_num == 0: 
                for model in classifiers:
                    logfiles[model] = open(f"./Logs/{model}/TestLogs/logs.csv", "w")
                    logfiles[model].write("BatchNum,Prediction,GroundTruth\n") # Clearing logs
                    logfiles[model].flush()
                    classifiers[model] = joblib.load(f"./Logs/{model}/final_model.sav")
            else:
                for model in classifiers:
                    logfiles[model] = open(f"./Logs/{model}/TestLogs/logs{model_num}.csv", "w")
                    logfiles[model].write("BatchNum,Prediction,GroundTruth\n") # Clearing logs
                    logfiles[model].flush()
                    classifiers[model] = joblib.load(f"./Logs/{model}/Models/{model_num}.sav")

        dstream.foreachRDD(lambda rdd: testBatch(rdd, cluster, model_num, clusterlogfile, logfiles))
    
    else: raise("Invalid argument to the argument mode of operation: Use '-m train' or '-m test'")

    ssc.start()            # Start the computation
    ssc.awaitTermination() # Wait for the computation to terminate
