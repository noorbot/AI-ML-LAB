from symbol import tfpdef
import pandas as pd
import numpy as np
from tqdm import tqdm

# def subsetDf(df, attribute1, attribute2, targVal):
#     return df.loc[:, [attribute1, attribute2, targVal]]

#Returns the norm of row
def calculateED(rowTrain, rowTest):
    elementSum = 0

    for index, x1 in rowTrain.items():
        x2 = rowTest.loc[index]
        elementSum += ((x1-x2)**2)

    dist = np.sqrt(elementSum)

    return  dist

def knn(testRow, trainDF, k):

    euclidDist = []
    trainLabels = trainDF.keys()
    trainTV = trainLabels[-1]
    #testRow = testRow.drop(labels=['Doggy'])

    for indexTrain, rowTrain in trainDF.iterrows():
        testClassification = rowTrain.iloc[-1]
        rowTrain = rowTrain.drop(labels=[trainTV])
        euclidDist.append(tuple([testClassification, calculateED(rowTrain,testRow)]))

    euclidDist.sort(key=lambda a: a[1])
    neighbourHood = []
    for i in range(k):
        neighbourHood.append(euclidDist[i])

    dfNeighbour = pd.DataFrame(neighbourHood)
    neighbour = dfNeighbour[0].value_counts().idxmax()

    return neighbour

# def findNeighbour(neighbour):
#     df = pd.DataFrame(neighbour)
#     counts = df[0].value_counts().idxmax()
#     return counts

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = knn(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

def evaluate(trainDF, testDF, k): # method to feed in test instances and calculate algorithm accuracy
    correct_predict = 0
    wrong_predict = 0

    testLabels = testDF.keys()
    testTV = testLabels[-1]

    for index, row in tqdm(testDF.iterrows()):
        rowQ = row
        testClassification = rowQ.iloc[-1]
        rowQ = rowQ.drop(labels=[testTV])
        neighbour = knn(rowQ, trainDF, k)

        if neighbour == testClassification:
            correct_predict += 1 #increase correct count
        else:
            wrong_predict += 1 #increase incorrect count

    accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy
    return accuracy

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
trainDataPath = trainData5
testDataPath = testData5

#For debugging and testing the program
# userDataPath = train
# testDataPath = test

#Create dataframes using csv files
trainDF = pd.read_csv(trainDataPath)
testDF = pd.read_csv(testDataPath)

#Obtain attributes of the data set and set target variable to last value of array
columnsNamesArr = trainDF.columns.values
targetAttribute = columnsNamesArr[-1]

k = 8;

#calculateED(trainDF, testDF)

# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]

dataset = [[2.7810836,2.550537003,0,'A'],
	[1.465489372,2.362125076,0,'A'],
	[3.396561688,4.400293529,0,'B'],
	[1.38807019,1.850220317,0,'B'],
	[3.06407232,3.005305973,0,'B'],
	[7.627531214,2.759262235,1,'A'],
	[5.332441248,2.088626775,1,'A'],
	[6.922596716,1.77106367,1,'B'],
	[8.675418651,-0.242068655,1,'B'],
	[7.673756466,3.508563011,1,'B']]

dataset2 = [[2.7810836,2.550537003,0,'A'],
	[1.465489372,2.362125076,0,'A'],
	[3.396561688,4.400293529,0,'B'],
	[1.38807019,1.850220317,0,'B'],
	[3.06407232,3.005305973,0,'B'],
	[7.627531214,2.759262235,1,'A'],
	[5.332441248,2.088626775,1,'A'],
	[6.922596716,1.77106367,1,'B'],
	[8.675418651,-0.242068655,1,'B'],
	[7.673756466,3.508563011,1,'B']]

dfTEST = pd.DataFrame(data=dataset, columns=['Alfred', 'Bobert', 'Charlie', 'Doggy'])
# row0 = dfTEST.iloc[0]
dfTEST2 = pd.DataFrame(data=dataset2,columns=['Alfred', 'Bobert', 'Charlie', 'Doggy'])
    
print('K = ', str(k), ', ', evaluate(trainDF, testDF, k))
