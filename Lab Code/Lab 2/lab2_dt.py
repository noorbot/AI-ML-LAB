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
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None

#Evaluate peformance of decision tree against test data
def evaluate(tree, testDF, label):
    correct_preditct = 0
    wrong_preditct = 0

    for index, row in testDF.iterrows(): #for each row in the dataset
        result = predict(tree, testDF.iloc[index]) #predict the row
        if result == testDF[label].iloc[index]: #predicted value and expected value is same or not
            correct_preditct += 1 #increase correct count
        else:
            wrong_preditct += 1 #increase incorrect count
    
    
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) #calculating accuracy
    return accuracy

def binDF(trainData, testData):
    return

trainData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\breast-cancer-wisconsin-wLabels-train.csv"
trainData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\letter-recognition-wLabels-train.csv"
trainData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\ecoli-wLabels-train.csv"
trainData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\agaricus-lepiota-wLabels-train.csv"
trainData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\lp5-formatted-wLabels-train.csv"

testData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\breast-cancer-wisconsin-wLabels-test.csv"
testData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\letter-recognition-wLabels-test.csv"
testData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\ecoli-wLabels-test.csv"
testData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\agaricus-lepiota-wLabels-test.csv"
testData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\lp5-formatted-wLabels-test.csv"

lecData = r"D:\Users\radam\Desktop\lecData.csv"


userDataPath = trainData3
trainDataPath = testData3
#userDataPath = lecData

#add header arguement to prevent first row from being read as labels. Enable whitespace delim line for whitespace delim datasets
df = pd.read_csv(userDataPath)
testDF = pd.read_csv(trainDataPath)

a = 5
b = 10

if userDataPath == trainData3:
    df['mcg'] = pd.cut(df['mcg'], bins = b)
    df['gvh'] = pd.cut(df['gvh'], bins = 20)
    df['aac'] = pd.cut(df['aac'], bins = a)
    df['alm1'] = pd.cut(df['alm1'], bins = b)
    df['alm2'] = pd.cut(df['alm2'], bins = a)

    testDF['mcg'] = pd.cut(testDF['mcg'], bins = b)
    testDF['gvh'] = pd.cut(testDF['gvh'], bins = 20)
    testDF['aac'] = pd.cut(testDF['aac'], bins = a)
    testDF['alm1'] = pd.cut(testDF['alm1'], bins = b)
    testDF['alm2'] = pd.cut(testDF['alm2'], bins = a)

    # df['mcg'] = pd.qcut(df['mcg'], q = a)
    # df['gvh'] = pd.qcut(df['gvh'], q = a)
    # df['aac'] = pd.qcut(df['aac'], q = a)
    # df['alm1'] = pd.qcut(df['alm1'], q = a)
    # df['alm2'] = pd.qcut(df['alm2'], q = a)

    # testDF['mcg'] = pd.qcut(testDF['mcg'], q = a)
    # testDF['gvh'] = pd.qcut(testDF['gvh'], q = a)
    # testDF['aac'] = pd.qcut(testDF['aac'], q = a)
    # testDF['alm1'] = pd.qcut(testDF['alm1'], q = a)
    # testDF['alm2'] = pd.qcut(testDF['alm2'], q = a)

    df = df.drop(['Sequence Name'], axis = 1)
    dtestDFf = testDF.drop(['Sequence Name'], axis = 1)

#elif userDataPath == trainData1:

columnsNamesArr = df.columns.values
targetAttribute = columnsNamesArr[-1]

# print(buildTree(df, targetAttribute))
tree = buildTree(df, targetAttribute)

#Evaluate Accuracy of Tree
accuracy = evaluate(tree, testDF, targetAttribute)
print("Accuracy of Decision Tree: " + str(accuracy*100) + "%")


