import pandas as pd
import numpy as np

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
    test = testRow.drop(labels=['Doggy'])

    for indexTrain, rowTrain in trainDF.iterrows():
        testClassification = rowTrain.iloc[-1]
        rowTrain = rowTrain.drop(labels=[trainTV])
        euclidDist.append(tuple([testClassification, calculateED(rowTrain,test)]))

    euclidDist.sort(key=lambda a: a[1])
    neighbours = []
    for i in range(k):
        neighbours.append(euclidDist[i])

    return neighbours

def findNeighbour(neighbour):
    df = pd.DataFrame(neighbour)
    counts = df[0].value_counts().idxmax()
    return counts

# def knn(trainDF, testDF, k):

#     euclidDist = []
#     # attributes = trainDF.keys()
#     # distance = 0

#     # testQRow = 0
    
#     # testQ = trainDF.iloc[testQRow]
#     #newDF = trainDF.drop([testQRow]).reset_index(drop=True)

#     #testRow = testDF.iloc[0]

#     for indexTest, rowTest in testDF.iterrows():
#         testRow = testDF.iloc[indexTest]
#         for indexTrain, rowTrain in trainDF.iterrows():
#             euclidDist.append(calculateED(testRow,rowTrain))

#     euclidDist.sort()
#     neighbours = []

#     for i in range(k):
#         neighbours.append(euclidDist[i])
#     # for index, row in trainDF.iterrows():
#     #     for index2, value in row.items():
#     #         print(f"Index : {index2}, Value : {value}")
#     #         distance += (value - awddwad)**2

#     #     euclidDist.update(distance*(1/2))

#     return neighbours

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = knn(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

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

dataset2 = [3.396561688,4.400293529,0,'B']

dfTEST = pd.DataFrame(data=dataset, columns=['Alfred', 'Bobert', 'Charlie', 'Doggy'])
# row0 = dfTEST.iloc[0]
dfTEST2 = pd.Series(dataset2,index=['Alfred', 'Bobert', 'Charlie', 'Doggy'])
    
# for index, row in dfTEST.iterrows():
#     print(calculateED(row0, row))
# testRow = dfTEST.iloc[0]
# testlabels = testRow.keys()
# print(testlabels[-1])
testNeigh = knn(dfTEST2, dfTEST, 3)
classifc = findNeighbour(testNeigh)
print(classifc)
