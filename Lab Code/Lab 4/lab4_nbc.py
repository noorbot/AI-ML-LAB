import pandas as pd
import numpy as np
from tqdm import tqdm

def calcPrior(trainDF, targVari):
    prior = []
    attData = trainDF[targVari].unique()

    for data in attData:
        subsetDF = trainDF[trainDF[targVari]==data]
        prior.append(len(subsetDF)/len(trainDF))

    return prior

def calcLikelihood(trainDF, attVari, attData, targVari, targData):
    subsetDF = trainDF[trainDF[targVari]==targData]
    likelihood = len(trainDF[trainDF[attVari]==attData]) / len(subsetDF)
    return likelihood

def naiveBayes(trainDF):
    attList = trainDF.keys()

    prior = calcPrior(trainDF, attList[-1])

    dataRowsList = trainDF.iloc[:,:-1].values

    for data in  dataRowsList:
        

    return

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
testData = r"D:\Users\radam\Desktop\test.csv"
test = r"D:\Users\radam\Desktop\wdbc_test_data.csv"
train = r"D:\Users\radam\Desktop\wdbc_train_data.csv"

#Set dataset to build decision tree from and test
# trainDataPath = trainData5
# testDataPath = testData5

#For debugging and testing the program
userDataPath = lecData
testDataPath = testData

#Debug dataframes using csv files
trainDF = pd.read_csv(userDataPath)
testDF = pd.read_csv(testDataPath)

#Create dataframes using csv files
# trainDF = pd.read_csv(trainDataPath)
# testDF = pd.read_csv(testDataPath)

#print(trainDF)

listOfAtt = trainDF.keys()

classes = sorted(list(trainDF['Outlook'].unique()))
classes2 = list(trainDF['Outlook'].unique())

test2 = trainDF['Outlook'].unique()

test = trainDF[trainDF['Play Tennis'] == 'No']

test_x = trainDF.iloc[:,:-1].values
test_y = trainDF.iloc[:,-1].values
print(test)

calcPrior(trainDF, listOfAtt[-1])