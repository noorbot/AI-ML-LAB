import pandas as pd
import numpy as np

# def subsetDf(df, attribute1, attribute2, targVal):
#     return df.loc[:, [attribute1, attribute2, targVal]]

#Returns the norm of row
def calculateED(rowTrain, rowTest):
    elementSum = 0

    for index, x1 in rowTrain.items():
        x2 = rowTest.iloc[index]
        elementSum += ((x1-x2)**2)

    dist = np.sqrt(elementSum)

    return  dist

def knn(trainDF, testDF, k):

    euclidDist = []
    # attributes = trainDF.keys()
    # distance = 0

    # testQRow = 0
    
    # testQ = trainDF.iloc[testQRow]
    #newDF = trainDF.drop([testQRow]).reset_index(drop=True)
    testRow = testDF.iloc[0]

    for index, row in trainDF.iterrows():
        #testRow = testDF.iloc[index]
        euclidDist.append(calculateED(testRow,row))

    euclidDist.sort()
    neighbours = []
    for i in range(k):
        neighbours.append(euclidDist[i])
    # for index, row in trainDF.iterrows():
    #     for index2, value in row.items():
    #         print(f"Index : {index2}, Value : {value}")
    #         distance += (value - awddwad)**2

    #     euclidDist.update(distance*(1/2))

    return neighbours


#File paths for training datasets
trainData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\wdbc_train_data.csv"
trainData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\letter-recognition-wLabels-train.csv"
trainData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\ecoli_train_data.csv"
trainData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\agaricus-lepiota-wLabels-train.csv"
trainData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\lp5_train_data.csv"

#File paths for testing datasets
testData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\wdbc_test_data.csv"
testData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\letter-recognition-wLabels-test.csv"
testData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\ecoli_test_data.csv"
testData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\agaricus-lepiota-wLabels-test.csv"
testData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\lp5_test_data.csv"

#Debugging datasets
lecData = r"D:\Users\radam\Desktop\lecData.csv"
test = r"D:\Users\radam\Desktop\wdbc_test_data.csv"
train = r"D:\Users\radam\Desktop\wdbc_train_data.csv"

#Set dataset to build decision tree from and test
userDataPath = trainData2
testDataPath = testData2

#For debugging and testing the program
# userDataPath = train
# testDataPath = test

#Create dataframes using csv files
trainDF = pd.read_csv(userDataPath)
testDF = pd.read_csv(testDataPath)

#Obtain attributes of the data set and set target variable to last value of array
columnsNamesArr = trainDF.columns.values
targetAttribute = columnsNamesArr[-1]

k = 2;

#calculateED(trainDF, testDF)

