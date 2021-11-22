# importing required libraries
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when


import numpy as np 
from sklearn.datasets import load_files

# Text cleaning and preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import HashingVectorizer

from tqdm.auto import tqdm

# from pyspark.ml import Pipeline
# from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
# from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
# from pyspark.ml.classification import LogisticRegression

RED = '\033[91m'
RESET = '\033[0m'
GREEN = '\033[92m'

# initializing spark session
sc = SparkContext(appName = "Spam Classifier")
spark = SparkSession(sc)
    
# Initializing the streaming context 
ssc = StreamingContext(sc, batchDuration = 3)

# Create a DStream that will connect to hostname:port, like localhost:9991
dstream = ssc.socketTextStream("localhost", 6100)


#tf = TfidfVectorizer()
hvec = HashingVectorizer(n_features = 2**9, alternate_sign = False) #***change this logically***

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

def preProcess(X, numRecords):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    corpus = []
    for i in tqdm(range(numRecords)):
        # Remove special symbols
        review = re.sub(r'\\r\\n', ' ', str(X[i]))
        # Remove all symbols except letters
        review = re.sub('[^a-zA-Z]', ' ', review)
        # Replacing all gaps with spaces 
        review = re.sub(r'\s+', ' ', review)                    
        # Remove 'b' in the beginning of each text
        review = re.sub(r'^b\s+', '', review)       

        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    
    # Creating the Bag of Words model
    X = hvec.fit_transform(corpus)
    return X.toarray()

batchNum = 1
def trainBatch(rdd):
  if not rdd.isEmpty():
    df = (
        spark.read.json(rdd, multiLine = True)
        .withColumn("data", F.explode(F.arrays_zip("feature0", "feature1", "feature2")))
        .select("data.feature0", "data.feature1", "data.feature2")
    )

    # print(RED)
    # df.printSchema()
    # print(RESET)

    # print(GREEN)
    # df.show(5)
    # print(RESET)

    X = preProcess(df.feature1, df.count())
    y = np.array(
        df.withColumn("feature2", when(col("feature2") == 'spam', 1).otherwise(0))
          .select("feature2")
          .collect()
    )

    # print(GREEN)
    # print(X)
    # print(y)
    # print(RESET)

    # Splitting data on train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=9, test_size=0.2)
    
    global model
    model = model.partial_fit(X_train, y_train, (0, 1))
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, pos_label = 1)
    recall = recall_score(y_test, pred, pos_label = 1)
    conf_m = confusion_matrix(y_test, pred)

    print(GREEN)
    print(f"accuracy: %.3f" %accuracy)
    print(f"precision: %.3f" %precision)
    print(f"recall: %.3f" %recall)
    print(f"confusion matrix: ")
    print(conf_m)
    print(RESET)

    with open("./TrainingLogs.txt", "a") as f:
        global batchNum
        f.write(f"Batch {batchNum}")
        f.write(f"\naccuracy: %.3f" %accuracy)
        f.write(f"\nprecision: %.3f" %precision)
        f.write(f"\nrecall: %.3f" %recall)
        f.write(f"\nconfusion matrix:\n")
        f.write(str(conf_m))
        f.write("\n-----------------------------------------------\n\n")
        batchNum += 1


dstream.foreachRDD(lambda rdd: trainBatch(rdd))

# Start the computation
ssc.start()             

# Wait for the computation to terminate
ssc.awaitTermination()  