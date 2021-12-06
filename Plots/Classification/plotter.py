import enum
from pyspark.sql import SparkSession
from pyspark.sql.functions import corr
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

def testingAccuracy(file,modelname):
    df = spark.read.csv(file,header=True,sep=",")
    pred = df.select('Prediction').rdd.map(
        lambda x: x[0]
    ).collect()

    gt = df.select('GroundTruth').rdd.map(
        lambda x: x[0]
    ).collect()

    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct+=1
    printI(f'{modelname} : {correct/len(gt)}')

    #1->spam 0->ham
    return correct/len(gt)


def graphManager(path_to_logs,mode=2): #2->training
    validDirs = {"Multi Layer Perceptron","Perceptron","SGD Classifier","Multinomial Naive Bayes"}
    attach=None

    testingAcc = []
    y = []

    if mode == 1: #testing
        attach = "TestLogs"
    else:
        attach = "TrainLogs"

    for dirname,subdirList,file in os.walk(path_to_logs):
        currFolder = dirname.split(os.sep)[-1] 

        if currFolder in validDirs:
            if mode==2:
                # csvPlotter(dirname + os.sep + attach + 'logs.csv',currFolder)
                printI(dirname + os.sep + attach + os.sep + 'logs.csv')
                csvPlotter(dirname + os.sep + attach + os.sep + 'logs.csv',currFolder)
            else:
                res = testingAccuracy(dirname + os.sep + attach + os.sep + 'logs.csv',currFolder)
                testingAcc.append(res)
                y.append(currFolder)
    
    if mode==1:

        fig = plt.figure(figsize = (10, 5)) 
        bars = plt.bar(y,testingAcc,width=0.4)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(),yval+0.005,int(yval*1000)/1000.)
            
        plt.show()


graphManager("./Logs",1)