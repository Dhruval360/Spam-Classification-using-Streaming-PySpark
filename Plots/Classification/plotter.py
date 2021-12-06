from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("plots").getOrCreate()
import matplotlib.pyplot as plt
import os

RED = '\033[91m'
RESET = '\033[0m'
GREEN = '\033[92m'

metrics={"Recall","Accuracy"}

def printI(toPrint):
    print(RED)
    print(toPrint)
    print(RESET)

def metricPlotter(xdata,metricName,df,modelname): 
    ydata = df.select(metricName).rdd.map(lambda x: x[0]).collect()

    plt.plot(xdata,ydata)
    plt.xlabel('batcNum')
    plt.ylabel(f'{metricName}')
    plt.title(f"{metricName} vs BatchSize for {modelname}")
    plt.show()

def csvPlotter(file,modelname):
    printI(file)

    df = spark.read.csv(file,header=True,sep=",")
    printI(df.show(5))
    
    batchNum = df.select('BatchNum').rdd.map(
        lambda x: x[0]
    ).collect()
    printI(batchNum)

    for a in metrics:
        metricPlotter(batchNum,a,df,modelname)

def graphManager(path_to_logs):
    validDirs = {"Multi Layer Perceptron","Perceptron","SGD Classifier"}
    for dirname,subdirList,file in os.walk(path_to_logs):
        currFolder = dirname.split(os.sep)[-1] 
        if currFolder in validDirs:
            csvPlotter(dirname + os.sep + 'logs.csv',currFolder)


graphManager("./TrainingLogs")