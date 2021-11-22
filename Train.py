from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import col, when, explode, arrays_zip

import re
import sys
import numpy as np 
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, Perceptron

# from pyspark.ml import Pipeline
# from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
# from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
# from pyspark.ml.classification import LogisticRegression

# Color coding the output to make it easier to find amongst the verbose output of Spark
RED = '\033[91m'
RESET = '\033[0m'
GREEN = '\033[92m'

hvec = HashingVectorizer(n_features = 2**9, alternate_sign = False)

models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'SGD Classifier': SGDClassifier(loss = 'log'),
    'Perceptron': Perceptron()
}

def preProcess(X, numRecords):
    stemmer = PorterStemmer()
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
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state = 9, test_size = 0.2)
    
    global models, batchNum
    
    for model in models:
        models[model] = models[model].partial_fit(X_train, y_train, (0, 1))
        pred = models[model].predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, pos_label = 1)
        recall = recall_score(y_test, pred, pos_label = 1)
        conf_m = confusion_matrix(y_test, pred)

        print(GREEN)
        print(f"Model = {model}")
        print(f"accuracy: %.3f" %accuracy)
        print(f"precision: %.3f" %precision)
        print(f"recall: %.3f" %recall)
        print(f"confusion matrix: ")
        print(conf_m)
        print(RESET)

        with open(f"./TrainingLogs/{model}/logs.txt", "a") as f:
            f.write(f"Batch {batchNum}")
            f.write(f"\naccuracy: %.3f" %accuracy)
            f.write(f"\nprecision: %.3f" %precision)
            f.write(f"\nrecall: %.3f" %recall)
            f.write(f"\nconfusion matrix:\n")
            f.write(str(conf_m))
            f.write("\n-----------------------------------------------\n\n")
        
    batchNum += 1

if __name__ == "__main__":
    # Initializing the spark session
    sc = SparkContext(appName = "Spam Classifier")
    spark = SparkSession(sc)
    spark.sparkContext.setLogLevel('WARN')

    # Initializing the streaming context 
    ssc = StreamingContext(
            sc, 
            batchDuration = (int(sys.argv[1]) if len(sys.argv) > 1 else 5) # Default batchDuration is 5s
          )

    # Create a DStream that will connect to hostname:port, like localhost:9991
    dstream = ssc.socketTextStream("localhost", 6100)
    dstream.foreachRDD(lambda rdd: trainBatch(rdd))

    ssc.start()            # Start the computation
    ssc.awaitTermination() # Wait for the computation to terminate