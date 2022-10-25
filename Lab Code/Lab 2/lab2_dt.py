import pandas as pd
import numpy as np
from tqdm import tqdm

#Calculate Total Entropy of Dataset
def entropyS(dataframe, column):
    df = dataframe
    entropy = 0
    targetCol = df.loc[:, column]
    instances = targetCol.value_counts()

    for x in instances:
        pi = x / len(targetCol)
        entropy += -pi*np.log2(pi)
    
    return entropy

#Calculate the entropy of the subset of the dataframe (b/w chosen atrribute and target variable)
def entropyAtt(df, attribute, targetVariable):
    targetCol = df.loc[: , targetVariable]
    attCol = df.loc[:, attribute]
    attColValues = attCol.unique()
    weightedFrac = 0
    entropyTotal = 0

    for attVal in attColValues:
        subdf = df[[attribute, targetVariable]]
        svSamples = subdf[subdf[attribute]==attVal]
        weightedFrac = len(svSamples)/len(subdf)
        entropyTotal += (-1)*weightedFrac*(entropyS(svSamples, targetVariable))
        

    return entropyTotal

#Calculate IG
def infoGain (entS, entSVTot):
    return entS + entSVTot

#Create subset df when partitioning data for next node of tree
def fiteredDf(df, attribute, val):
    return df[df[attribute] == val].reset_index(drop=True)

#Find Winner of IG Array
def findWinner(list):
    attributeWin = max(list, key=list.get)
    return attributeWin

#Recursive function to build tree
def buildTree(df,targetVariable, tree=None): 
    allAtt = df.columns.values  
    targetVariable = targetVariable
    
    #find the stump
    entropyTotal = entropyS(df, targetVariable)
    ig = {}

    for column in allAtt[:-1]:
        entropyDF = entropyAtt(df, column , targetAttribute)
        ig.update({column:infoGain(entropyTotal, entropyDF)})

    if len(ig) == 0:
        return tree
    else:
        node = findWinner(ig)
    # print(ig)
    # node = findWinner(ig)
   

    # #Get uniques of winner attribute
    nodeCol = df.loc[:, node]
    attColValues = nodeCol.unique()

    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    

    #For loop to create ID3 Decision Tree Recursivley
    for value in attColValues:
        
        filteredDF = fiteredDf(df,node,value)
        subDF = filteredDF.drop([node], axis = 1)

        #print(value)
        
        clValue,counts = np.unique(subDF[targetVariable],return_counts=True)
            
        #Check Purity of Sub DF
        if len(counts)==1:
            tree[node][value] = clValue[0]

        #Recursive Call                                                 
        else:        
            tree[node][value] = buildTree(subDF, targetVariable) 
                   
    return tree

#Predict tree from test data
def predict(tree, instance):

    #Check if it is a leaf node
    if not isinstance(tree, dict):
        return tree
    else:
        #get attribute name from dictionary
        root_node = next(iter(tree))
        #value of the feature
        feature_value = instance[root_node] 
        #checking the feature value in current tree node
        if feature_value in tree[root_node]: 
            #goto next feature
            return predict(tree[root_node][feature_value], instance) 
        else:
            return None

#Evaluate peformance of decision tree against test data
def evaluate(tree, testDF, label):
    correct_preditct = 0
    wrong_preditct = 0

    #Iterate through rows of dataframe
    for index, row in testDF.iterrows(): 

        #predict the row
        result = predict(tree, testDF.iloc[index]) 

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

#Bin datasets for attributes with continuous data
def binDF(df, testDF, userDataPath):

    if userDataPath == trainData3:
        df['mcg'] = pd.cut(df['mcg'], bins = 20)
        df['gvh'] = pd.cut(df['gvh'], bins = 20)
        df['aac'] = pd.cut(df['aac'], bins = 20)
        df['alm1'] = pd.cut(df['alm1'], bins = 20)
        df['alm2'] = pd.cut(df['alm2'], bins = 20)

        testDF['mcg'] = pd.cut(testDF['mcg'], bins = 20)
        testDF['gvh'] = pd.cut(testDF['gvh'], bins = 20)
        testDF['aac'] = pd.cut(testDF['aac'], bins = 20)
        testDF['alm1'] = pd.cut(testDF['alm1'], bins = 20)
        testDF['alm2'] = pd.cut(testDF['alm2'], bins = 20)

    df = df.drop(['Sequence Name'], axis = 1)
    dtestDFf = testDF.drop(['Sequence Name'], axis = 1)

    return df, testDF

#elif userDataPath == trainData1:
    return

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
userDataPath = trainData1
testDataPath = testData1

#For debugging and testing the program
#userDataPath = lecData
# userDataPath = train
# testDataPath = test

#Create dataframes using csv files
df = pd.read_csv(userDataPath)
testDF = pd.read_csv(testDataPath)

#Obtain attributes of the data set and set target variable to last value of array
columnsNamesArr = df.columns.values
targetAttribute = columnsNamesArr[-1]

#Use training Data to create tree and print developed tree to terminal
print(buildTree(df, targetAttribute))
tree = buildTree(df, targetAttribute)

#Evaluate Accuracy of Tree
accuracy = evaluate(tree, testDF, targetAttribute)
print("Accuracy of Decision Tree: " + str(accuracy*100) + "%")


