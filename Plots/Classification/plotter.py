from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("plots").getOrCreate()
import matplotlib.pyplot as plt
import os
import sys

RED = '\033[91m'
RESET = '\033[0m'
GREEN = '\033[92m'

metrics={"Recall","Accuracy","Precision"}

modelWiseData = {"Multi Layer Perceptron":[],"Perceptron":[],
"SGD Classifier":[],"Multinomial Naive Bayes":[]}

def printI(toPrint):
    print(RED)
    print(toPrint)
    print(RESET)

def metricPlotter(xdata,metricName,df,modelname): 
    ydata = df.select(metricName).rdd.map(lambda x: x[0]).collect()
    ydata = [float(i) for i in ydata]

    plt.plot(xdata,ydata)
    plt.xlabel('batches')
    plt.ylabel(f'{metricName}')
    plt.title(f"{metricName} vs Batches for {modelname}")
    plt.show()

def csvPlotter(file,modelname):
    printI(file)

    df = spark.read.csv(file,header=True,sep=",")
    # printI(df.show(5))
    
    batchNum = df.select('Batches').rdd.map(
        lambda x: x[0]
    ).collect()
    # printI(batchNum)
    batchNum = [int(i) for i in batchNum]

    for a in metrics:
        metricPlotter(batchNum,a,df,modelname)

def testingAccuracy(file,modelname):
    df = spark.read.csv(file,header=True,sep=",")
    pred = df.select('Prediction').rdd.map(
        lambda x: x[0]
    ).collect()
    pred = [int(i) for i in pred]

    gt = df.select('GroundTruth').rdd.map(
        lambda x: x[0]
    ).collect()
    gt = [int(i) for i in gt]

    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct+=1
    printI(f'{modelname} : {correct/len(gt)}')

    #1->spam 0->ham
    return correct/len(gt)

def perModelAccum(file,modelname):
    df = spark.read.csv(file,header=True,sep=",")
    
    for a in metrics:
        f  = df.select(f'{a}').rdd.map(
            lambda x: x[0]
        ).collect()

        global modelWiseData
        # print(modelname in modelWiseData.keys())
        # print(modelWiseData[modelname])
        # if a not in modelWiseData[modelname]:
        #     modelWiseData[modelname].append({a:f})
        # else:
        #     printI("What!!??")
        modelWiseData[modelname].append({a:f})

def getXdata(file):
    df = spark.read.csv(file,header=True,sep=",")
    printI(df.show(5))
    
    batchNum = df.select('BatchNum').rdd.map(
        lambda x: x[0]
    ).collect()

    
    return [int(i) for i in batchNum]

def allMetricPerModelPlotter(xdata):
    fig = plt.figure(figsize = (50, 50)) 

    for metric in metrics:
        for model,allmetrics in modelWiseData.items():
            # printI(allmetrics)
            for iter,ametric in enumerate(allmetrics):
                if list(ametric.keys())[0]==metric:
                    i=iter
            #print(allmetrics[i][metric])
            y_data = [float(j) for j in allmetrics[i][metric]]
            plt.plot(xdata,y_data,label=model)
        plt.title(f"{metric} vs batches")
        plt.legend()
        plt.show()



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
            if mode==2:#training
                # csvPlotter(dirname + os.sep + attach + 'logs.csv',currFolder)
                printI(dirname + os.sep + attach + os.sep + 'logs.csv')
                csvPlotter(dirname + os.sep + attach + os.sep + 'logs.csv',currFolder)

            elif mode==1: #testing
                printI(dirname + os.sep + attach + os.sep + 'logs.csv')
                res = testingAccuracy(dirname + os.sep + attach + os.sep + 'logs.csv',currFolder)
                testingAcc.append(res)
                y.append(currFolder)
            else: #3 
                printI(dirname + os.sep + attach + os.sep + 'logs.csv')
                printI(currFolder)
                perModelAccum(dirname + os.sep + attach + os.sep + 'logs.csv',currFolder)
    
    if mode==1:

        fig = plt.figure(figsize = (50,50)) 
        bars = plt.bar(y,testingAcc,width=0.4)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(),yval+0.005,int(yval*1000)/1000.)
            
        plt.show()

    elif mode==3: #per model accumulate
        allMetricPerModelPlotter(getXdata('./Logs/Multi Layer Perceptron/TrainLogs/logs.csv'))

'''
for testing logs plots - 1
for traing log plots - 2
for per metric graph - 3
'''
# graphManager("./Logs",1)
#graphManager("./Logs",3)
# graphManager("./Logs",2)

if __name__ == "__main__":
    graphManager(sys.argv[1],int(sys.argv[2]))