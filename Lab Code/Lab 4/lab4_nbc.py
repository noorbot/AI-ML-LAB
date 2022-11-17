import pandas as pd
import numpy as np
from tqdm import tqdm

#Function to calculate prior probability of dataset P(vj)
def calcPrior(trainDF, targVari):
    #Create a dictionary to store the P(vj) of the target variable. Used to refer to the correct classifcation when finding max posterior value.
    prior = {}

    #Get all unique values of target variable
    attData = trainDF[targVari].unique()

    #Iterate through unique values, calculate P(vj), and store in dictionary with the associated classifcation key
    for data in attData:
        subsetDF = trainDF[trainDF[targVari]==data]
        prior.update({data:(len(subsetDF)/len(trainDF))})

    return prior

#Function to calculate the likelihood of one attribute - P(ai|vj)
def calcLikelihood(trainDF, attVari, attData, targVari, targData):
    #Create a filtered subset that only accounts for specifc attribute and attribute value to be calculated
    subsetDF = trainDF[trainDF[targVari]==targData]

    #Calculate likelihood of specifc attribute value and target variable P(ai|vj)
    likelihood = len(subsetDF[subsetDF[attVari]==attData]) / len(subsetDF)

    return likelihood

#Function to launch naive bayes classifier
def naiveBayes(trainDF, testDF):
    #Get all attribute names in the dataframe
    attList = trainDF.keys()

    #Store all unqiue values of target variable
    targData = trainDF[attList[-1]].unique()

    #Get all row values of the test dataset, excluding the test variable and store this as a list
    testRows = testDF.iloc[:,:-1].values

    #Get all target variable values in the test dataset and store this as a list
    testValues = testDF[attList[-1]].tolist()

    #Initialize the prediction list for later use
    prediction = []
    
    #Calculate the P(vj) of the entire training dataset
    prior = calcPrior(trainDF,attList[-1])

    #Loop through all test rows of the training dataset. Nested for loops are used to pass a single attribute of a specifc row instance into calcLikelihood()
    #This is then multiplied with the subsequent likelihoods for the given row instance. After the calculation of the liklihood of the whole row is calculated
    #The prior probablity is then multipled to the correct likelihoods by passing targData[i] as a key for the prior dictionary. This is then appended as the
    #posterior probability. After a single row instance is finished, the variables are reset to take in the next row instance.
    #Implemented TQDM properly compared to Lab 3 to show a progress bar when executing the program
    for testInst in tqdm(testRows, desc="Naive Bayes Classifier - Dataset"):
        posterior = []
        for i in range (len(targData)):
            liklihoodInst = 1
            for j in range (len(attList) -1):
                liklihoodInst *= calcLikelihood(trainDF, attList[j], testInst[j], attList[-1], targData[i])
            posterior.append(tuple([targData[i],liklihoodInst*prior[targData[i]]]))

        #Find the max posterior probability for a given row instance. Store the tuple winner into the prediction list
        prediction.append(max(posterior, key=lambda tup: tup[1])) 

    #Check accuracy of Naive Bayes Classifier, initialize correct and wrong count variables
    correct_preditct = 0
    wrong_preditct = 0
  
    #Iterate through test data set classifcation
    for i in range (len(testValues)):
        #Check if predicted value is the same a value from the produced prediction list
        if prediction[i][0] == testValues[i]:
        #increase correct count
            correct_preditct += 1 
        else:
         #increase wrong count
            wrong_preditct += 1 
    
    #Calculate the accuracy of Naive Bayes Classifer and print to terminal
    accuracy = correct_preditct / (correct_preditct + wrong_preditct)
    print(accuracy)

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
trainDataPath = trainData5
testDataPath = testData5

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

naiveBayes(trainDF, testDF)