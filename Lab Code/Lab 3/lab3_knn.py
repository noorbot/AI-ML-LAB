import pandas as pd
import numpy as np
from tqdm import tqdm

#Returns the norm of row
def calculateED(rowTrain, rowTest):
    elementSum = 0

    #Using both rows passed to function, iterate through series and calculate the ED
    for index, x1 in rowTrain.items():
        x2 = rowTest.loc[index]
        elementSum += ((x1-x2)**2)

    dist = np.sqrt(elementSum)

    return  dist

#Peform KNN
def knn(testRow, trainDF, k):

    euclidDist = []
    trainLabels = trainDF.keys()
    trainTV = trainLabels[-1]

#Iterate through training dataset and store ED in a tuple
    for indexTrain, rowTrain in trainDF.iterrows():
        testClassification = rowTrain.iloc[-1]
        rowTrain = rowTrain.drop(labels=[trainTV])
        euclidDist.append(tuple([testClassification, calculateED(rowTrain,testRow)]))

#Sort list of tuples in accending order and store the K values in the list neighbourHood
    euclidDist.sort(key=lambda a: a[1])
    neighbourHood = []
    for i in range(k):
        neighbourHood.append(euclidDist[i])

#Using the list of tuples with k values, find the classifcation with the most votes
    dfNeighbour = pd.DataFrame(neighbourHood)
    neighbour = dfNeighbour[0].value_counts().idxmax()

    return neighbour

#Function to start KNN algorithim
def predictNeighbour(trainDF, testDF, k): # method to feed in test instances and calculate algorithm accuracy
    correct_predict = 0
    wrong_predict = 0

    testLabels = testDF.keys()
    testTV = testLabels[-1]

#Iterate through testing dataset and pass each row as a test query to KNN
    for index, row in tqdm(testDF.iterrows()):
        rowQ = row
        testClassification = rowQ.iloc[-1]
        rowQ = rowQ.drop(labels=[testTV])
        neighbour = knn(rowQ, trainDF, k)

#Check resutling prediction against true value of test query and increment accoridngly
        if neighbour == testClassification:
            correct_predict += 1
        else:
            wrong_predict += 1

#Calculate the accuacy of KNN
    accuracy = correct_predict / (correct_predict + wrong_predict)
    print('K = ', str(k), ', ', accuracy)

#File paths for training datasets
trainData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\wdbc_train_data.csv"
trainData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\letter-recognition-wLabels-train.csv"
trainData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\ecoli_train_data.csv"
trainData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\agaricus-lepiota-wLabels-train.csv"
trainData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\lp5_train_data.csv"

#File paths for testing datasets
testData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\wdbc_test_data.csv"
testData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\letter-recognition-wLabels-test_400.csv"
testData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\ecoli_test_data.csv"
testData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\agaricus-lepiota-wLabels-test_400.csv"
testData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\lp5_test_data.csv"

#Debugging datasets
lecData = r"D:\Users\radam\Desktop\lecData.csv"
test = r"D:\Users\radam\Desktop\wdbc_test_data.csv"
train = r"D:\Users\radam\Desktop\wdbc_train_data.csv"

#Set dataset to build decision tree from and test
trainDataPath = trainData5
testDataPath = testData5

#For debugging and testing the program
# userDataPath = train
# testDataPath = test

#Create dataframes using csv files
trainDF = pd.read_csv(trainDataPath)
testDF = pd.read_csv(testDataPath)

#Set desired K value
k = 5; 

#Start KNN
predictNeighbour(trainDF, testDF, k)