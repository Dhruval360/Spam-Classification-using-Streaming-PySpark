# Spam Classification using Streaming PySpark
A project that involves collecting streaming email data over a TCP socket, and training multiple spam classifiers on it in real-time. Project was implemented in PySpark, tested on version 3.1.2

This project was submitted as part of the final project of the Big Data course (UE19CS322) at PES University, Bengaluru by the team `BD_123_272_313_393`
#### Authors 
[Chandradhar Rao](https://github.com/chandradharrao) , [Dhruval PB](https://github.com/Dhruval360) , [Mihir M Kestur](https://github.com/mihirkestur)

## Project Structure
```
Machine-Learning-with-Spark-Streaming
├── Logs
│   ├── Clustering                      
│   │   ├── final_model.sav          [Final Model obtained after training over the entire dataset]
│   │   ├── Models                   [Models saved after training over each batch of the dataset]                           
│   │   ├── TestLogs                 [Log files containing the testing metrics]                   
│   │   └── TrainLogs                [Log files containing the training metrics]
│   ├── Multi Layer Perceptron          
│   │   ├── final_model.sav
│   │   ├── Models 
│   │   ├── TestLogs
│   │   └── TrainLogs
│   ├── Multinomial Naive Bayes
│   │   ├── final_model.sav
│   │   ├── Models 
│   │   ├── TestLogs
│   │   └── TrainLogs
│   ├── Perceptron
│   │   ├── final_model.sav
│   │   ├── Models 
│   │   ├── TestLogs
│   │   └── TrainLogs
│   └── SGD Classifier
│       ├── final_model.sav
│       ├── Models 
│       ├── TestLogs
│       └── TrainLogs
├── README.md
├── spam                              [The dataset]
│   ├── test.csv
│   └── train.csv
├── Stream.py                         [TCP Server that serves batches of the dataset as JSONs]
├── Train.py                          [PySpark program to train and test the aforementioned models]
├── Plot.ipynb                        [Notebook containing the analysis of the training and testing metrics obtained with batch size 100]
├── Plot-200.ipynb                    [Notebook containing the analysis of the training and testing metrics obtained with batch size 200]
└── Plotter.py                        [Used for the analysis of the training and testing metrics obtained (using PySpark)]
```

## Dependencies
- Spark 3.1.2 (PySpark comes with the Spark installation)
- Python 3 interpreter with all the packages in `Requirements.txt` installed

## Installing the dependencies:

```bash
$ python3 -m pip install -r Requirements.txt
$ python3 -c "exec(\"import nltk\nnltk.download('stopwords')\")" ## Downloading stopwords
```
## Running the program on a single Spark worker


Run the following command to print the help doc for the programs:
```bash
$ python3 Stream.py --help
$ spark-submit Train.py --help  ## This assumes spark has been added to path
```

Run the `Stream.py` program in one terminal (or in the background) and `Train.py` (using spark-submit) in another for training the aforementioned spam classifiers on the dataset provided in the repo.

## Running the program on multiple Spark workers
```
Coming soon...
```
