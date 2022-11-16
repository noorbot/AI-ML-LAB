import pandas as pd
import numpy as np
from tqdm import tqdm

def calcPrior(trainDF, targVari):
    prior = {}
    # prior = []
    attData = trainDF[targVari].unique()

    for data in attData:
        subsetDF = trainDF[trainDF[targVari]==data]
        # prior.append(len(subsetDF)/len(trainDF))
        prior.update({data:(len(subsetDF)/len(trainDF))})

    return prior

def calcLikelihood(trainDF, attVari, attData, targVari, targData):
    subsetDF = trainDF[trainDF[targVari]==targData]
    likelihood = len(subsetDF[subsetDF[attVari]==attData]) / len(subsetDF)
    return likelihood

def naiveBayes(trainDF, testRow):
    likelihoodTotal = []
    posterior = []
    attList = trainDF.keys()
    targData = trainDF[attList[-1]].unique()
    calculatedTuple = 1

    prior = calcPrior(trainDF, attList[-1])

    #dataRowsList = trainDF.iloc[:,:-1].values

    # dataRowsDF = trainDF.drop([listOfAtt[-1]], axis=1)

    # for indexTrain, rowTrain in dataRowsDF.iterrows():
    #     dataRowsTuples = list(rowTrain.items())

    # for indexTrain, rowTrain in testDF.iterrows():
    #     dataRowsTuples = list(rowTrain.items())
    #     likelihood = []
    #     calculatedTuple = 1
    #     for i in range ((len(dataRowsTuples) -1)):
    #         calculatedTuple *= calcLikelihood(trainDF, dataRowsTuples[i][0], dataRowsTuples[i][1], attList[-1], trainDF.loc[indexTrain, attList[-1]])
    #         #likelihoodTotal.append(tuple([dataRowsTuples[i][1],trainDF.loc[indexTrain, attList[-1]], calculatedTuple]))
    #     #likelihoodTotal.append(tuple([indexTrain, trainDF.loc[indexTrain, attList[-1]], calculatedTuple]))
    #     likelihoodTotal.append(tuple([trainDF.loc[indexTrain, attList[-1]], calculatedTuple*prior[trainDF.loc[indexTrain, attList[-1]]]]))      
    
    dataRowsTuples = list(testRow.items())
    calculatedTuple = 1
    for i in range ((len(dataRowsTuples) -1)):
        calculatedTuple *= calcLikelihood(trainDF, dataRowsTuples[i][0], dataRowsTuples[i][1], attList[-1], dataRowsTuples[-1][-1])



            #likelihoodTotal.append(tuple([dataRowsTuples[i][1],trainDF.loc[indexTrain, attList[-1]], calculatedTuple]))
        #likelihoodTotal.append(tuple([indexTrain, trainDF.loc[indexTrain, attList[-1]], calculatedTuple])
    likelihoodTotal.append(tuple([dataRowsTuples[-1][-1], calculatedTuple*prior[dataRowsTuples[-1][-1]]])) 
    # for i in range ((len(likelihoodTotal))):
    #     posterior_value = likelihoodTotal[i][1] * prior[likelihoodTotal[i][0]]
    #     posterior.append(tuple([likelihoodTotal[i][0], posterior_value]))

    # print(dataRowsTuples)
    # # for i in range (len(dataRowsTuples)):
    # #     print(dataRowsTuples[0][0])
    # for k, v in dataRowsTuples:
    #     print(k, v)
    # posterior.sort(key=lambda a: a[1])
    # max_tuple = max(posterior, key=lambda tup: tup[1])



    max_tuple = max(likelihoodTotal, key=lambda tup: tup[1])
    return max_tuple[0]

def evaluate(trainDF, testDF):
    correct_preditct = 0
    wrong_preditct = 0
    listOfAtt = testDF.keys()
    label = listOfAtt[-1]
    #Iterate through rows of dataframe
    for index, row in tqdm(testDF.iterrows()): 

        #predict the row
        result = naiveBayes(trainDF, row) 

        #Check if predicted value is the same a value from built tree
        if result == testDF[label].iloc[index]:
            #increase correct count
            correct_preditct += 1 
        else:
            #increase incorrect count
            wrong_preditct += 1 
    
    #Calculate the accuracy of trained tree
    accuracy = correct_preditct / (correct_preditct + wrong_preditct)
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
lecDataTest = r"D:\Users\radam\Desktop\lecDataTest.csv"
testData = r"D:\Users\radam\Desktop\test.csv"
test = r"D:\Users\radam\Desktop\wdbc_test_data.csv"
train = r"D:\Users\radam\Desktop\wdbc_train_data.csv"

#Set dataset to build decision tree from and test
trainDataPath = trainData3
testDataPath = testData3

#For debugging and testing the program
# userDataPath = lecData
# #testDataPath = testData
# testDataPath= lecDataTest

#Debug dataframes using csv files
# trainDF = pd.read_csv(userDataPath)
# testDF = pd.read_csv(testDataPath)

#Create dataframes using csv files
trainDF = pd.read_csv(trainDataPath)
testDF = pd.read_csv(testDataPath)

#print(trainDF)

# listOfAtt = trainDF.keys()

# classes = sorted(list(trainDF['Outlook'].unique()))
# classes2 = list(trainDF['Outlook'].unique())

# test2 = trainDF['Outlook'].unique()

# test = trainDF[trainDF['Play Tennis'] == 'No']

# test_x = trainDF.iloc[:,:-1].values
# test_y = trainDF.iloc[:,-1].values
# #print(test_x)

# #print(calcPrior(trainDF, listOfAtt[-1]))
# subsetDF = trainDF.drop([listOfAtt[-1]], axis=1)


print(evaluate(trainDF, testDF))



# print(calcLikelihood(trainDF, 'Outlook', 'Sunny', 'Play Tennis', 'No'))
# print(calcLikelihood(trainDF, 'Outlook', 'Sunny', 'Play Tennis', 'Yes'))
