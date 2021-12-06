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
RED = '\033[91m'
RESET = '\033[0m'
GREEN = '\033[92m'

#tokens are encded as numerical indexes using hash function but once hashed,multiple tokens can map to same index so they cannot be retrieved.
#2**9 n_features means the number of feature columns is 2**9.
hvec = HashingVectorizer(n_features = 2**9, alternate_sign = False)
stemmer = PorterStemmer() # Stems the words. Eg: Converts running, ran, run to run.

'''
Multinomial Naive Bayes:Bayesian classifier for discrete features.
alpha -> laplace smoothing

'''
classifiers = { # TODO: Explain Models
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
    # re.compile(r'^b\s+'),     # Select Word boundaries before consecutive spaces
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
    Reads a JSON rdd into a DataFrame, performs preprocessing
    Returns Features X and Labels y
    '''
    global hvec
    df = (
            spark.read.json(rdd, multiLine = True)
            .withColumn("data", explode(arrays_zip("feature0", "feature1", "feature2"))) # TODO: Check what explode does
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
        X, y = readStream(rdd)

        global classifiers, batchNum

        for model in classifiers:
            classifiers[model] = classifiers[model].partial_fit(X, y.reshape((len(y),)), np.unique(y)) # Why reshape
            pred = classifiers[model].predict(X)

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
        X, y = readStream(rdd)

        global clustering_model, batchNum

        clustering_model = clustering_model.partial_fit(X) # Why reshape
        pred = clustering_model.predict(X)

        accuracy_spam_0 = accuracy_score(y, pred)
        precision_spam_0 = precision_score(y, pred, labels = np.unique(y))
        recall_spam_0 = recall_score(y, pred, labels = np.unique(y))
        conf_m_spam_0 = confusion_matrix(y, pred)

        print(GREEN)
        print(f"Model = Clustering")
        
        print(f"MEASURES WHEN SPAM IS ENCODED AS 1")
        print(f"accuracy: %.3f" %accuracy_spam_0)
        print(f"precision: %.3f" %precision_spam_0)
        print(f"recall: %.3f" %recall_spam_0)
        print(f"confusion matrix: ")
        print(conf_m_spam_0)
        
        print(RESET)
        print(RED)

        y = [~i for i in y] 

        print(f"MEASURES WHEN SPAM IS ENCODED AS 1")

        accuracy_spam_1 = accuracy_score(y, pred)
        precision_spam_1 = precision_score(y, pred, labels = np.unique(y))
        recall_spam_1 = recall_score(y, pred, labels = np.unique(y))
        conf_m_spam_1 = confusion_matrix(y, pred)

        print(f"accuracy: %.3f" %accuracy_spam_1)
        print(f"precision: %.3f" %precision_spam_1)
        print(f"recall: %.3f" %recall_spam_1)
        print(f"confusion matrix: ")
        print(conf_m_spam_1)


        print("\n\nSaving Model to disk...")

        joblib.dump(classifiers[model], f"./Logs/Clustering/Models/{batchNum}.sav") # sav?
        joblib.dump(classifiers[model], f"./Logs/Clustering/final_model.sav")
        print("Model saved to disk")
        print(RESET)
        
        with open(f"./Logs/Clustering/TrainLogs/logs.csv", "a") as f:
            f.write(f"{batchNum},{accuracy_spam_0},{precision_spam_0},{recall_spam_0},{accuracy_spam_1},{precision_spam_1},{recall_spam_1}\n")

        batchNum += 1

numBatches = None


'''
Input:
    rdd: The test rdd upon which the model to run
    model_num: Number of the model to be chosen for which testing is to happen. Default is the final_model.sav that is saved
Output:
    Logs the information [batch number, prediction value, actual value] into the csv file in TrainLogs for each classifier
'''
batchNum = 1
def testBatch(rdd, model_num = None, cluster = 0):
    if not rdd.isEmpty():
        
        global batchNum

        # Read stream and pre-process it
        X, gt_values = readStream(rdd)
        
        if(cluster == 0):
            # For all the classifiers load the right model, predict on the test rdd (i.e. X), Log to file
            for model in classifiers:

                if(model_num is not None):
                    final_model = joblib.load(f"./Logs/{model}/final_model.sav")
                else:
                    final_model = joblib.load(f"./Logs/{model}/Models/{model_num}.sav")
                
                prediction = final_model.predict(X)

                print(GREEN)
                print(f"Model = {model}")

                with open(f"./Logs/{model}/TestLogs/logs.csv", "a") as f:
                    for i in zip(prediction, gt_values):
                        f.write(f"{batchNum},{i[0]},{i[1][0]}\n")

                print(f"Successfully logged to file...\n")
                print(RESET)
            f.close()
        else:
            if(model_num is not None):
                final_model = joblib.load(f"./Logs/Clustering/final_model.sav")
            else:
                final_model = joblib.load(f"./Logs/Clustering/Models/{model_num}.sav")
            
            prediction = final_model.predict(X)

            print(GREEN)
            print(f"Model = Clustering")

            with open(f"./Logs/Clustering/TestLogs/logs.csv", "a") as f:
                for i in zip(prediction, gt_values):
                    f.write(f"{batchNum},{i[0]},{i[1][0]}\n")

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

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    batchDuration = args.delay
    mode = args.mode
    cluster = args.clustering

    if(args.clean): # Clear all logs and start afresh
        for model in classifiers:
            f = open(f"./Logs/{model}/TrainLogs/logs.csv", "w")
            f.write("BatchNum,Accuracy,Precision,Recall\n")

            f = open(f"./Logs/{model}/TestLogs/logs.csv", "w")
            f.write("BatchNum,Prediction,GroundTruth\n")
            
            f.close()
        
        f = open(f"./Logs/Clustering/TrainLogs/logs.csv", "w")
        f.write("batchNum,accuracy_spam_0,precision_spam_0,recall_spam_0,accuracy_spam_1,precision_spam_1,recall_spam_1\n")
        
        f = open(f"./Logs/Clustering/TestLogs/logs.csv", "w")
        f.write("BatchNum,Prediction,GroundTruth\n")

        print(f"\n{GREEN}Cleaned the Logs{RESET}\n")
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
        dstream.foreachRDD(lambda rdd: testBatch(rdd, cluster))
    else: raise("Invalid argument to the argument mode of operation: Use '-m train' or '-m test'")

    ssc.start()            # Start the computation
    ssc.awaitTermination() # Wait for the computation to terminate
