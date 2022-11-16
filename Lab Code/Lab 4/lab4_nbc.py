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

def naiveBayes(trainDF, testDF):
    attList = trainDF.keys()
    targData = trainDF[attList[-1]].unique()

    
    testRows = testDF.iloc[:,:-1].values
    testValues = testDF[attList[-1]].tolist()
    prediction = []
    
    prior = calcPrior(trainDF,attList[-1])

    for testInst in testRows:
        posterior = []
        for i in range (len(targData)):
            liklihoodInst = 1
            for j in range (len(attList) -1):
                liklihoodInst *= calcLikelihood(trainDF, attList[j], testInst[j], attList[-1], targData[i])
            #     print(liklihoodInst)
            # print('DONE')
            posterior.append(tuple([targData[i],liklihoodInst*prior[targData[i]]]))
        prediction.append(max(posterior, key=lambda tup: tup[1])) 



    correct_preditct = 0
    wrong_preditct = 0
  

    for i in range (len(testValues)):
        #Check if predicted value is the same a value from built tree
        if prediction[i][0] == testValues[i]:
        #increase correct count
            correct_preditct += 1 
        else:
         #increase incorrect count
            wrong_preditct += 1 
    
    #Calculate the accuracy of trained tree
    accuracy = correct_preditct / (correct_preditct + wrong_preditct)
    print(accuracy)


    return 

# def evaluate(trainDF, testDF):
#     correct_preditct = 0
#     wrong_preditct = 0
#     listOfAtt = testDF.keys()
#     label = listOfAtt[-1]
#     #Iterate through rows of dataframe


#     for index, testRow in tqdm(testDF.iterrows()): 
#         #predict the row
#         result = piLikelihood(trainDF, testRow) 
#         #Check if predicted value is the same a value from built tree
#         if result == testDF[label].iloc[index]:
#             #increase correct count
#             correct_preditct += 1 
#         else:
#             #increase incorrect count
#             wrong_preditct += 1 
    
#     #Calculate the accuracy of trained tree
#     accuracy = correct_preditct / (correct_preditct + wrong_preditct)
#     print(accuracy)

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
# trainDataPath = trainData5
# testDataPath = testData5

#For debugging and testing the program
userDataPath = lecData
#testDataPath = testData
testDataPath= lecDataTest

#Debug dataframes using csv files
trainDF = pd.read_csv(userDataPath)
testDF = pd.read_csv(testDataPath)

#Create dataframes using csv files
# trainDF = pd.read_csv(trainDataPath)
# testDF = pd.read_csv(testDataPath)

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

testDF = trainDF

naiveBayes(trainDF, testDF)



# print(calcLikelihood(trainDF, 'Outlook', 'Sunny', 'Play Tennis', 'No'))
# print(calcLikelihood(trainDF, 'Outlook', 'Sunny', 'Play Tennis', 'Yes'))
