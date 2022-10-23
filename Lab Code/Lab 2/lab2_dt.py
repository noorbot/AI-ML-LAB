import pandas as pd
import numpy as np

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

#Find Winner
def findWinner(list):
    attributeWin = max(list, key=list.get)
    return attributeWin

def createSubTree(df, attribute, targetVariable):
    return

inputData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\breast-cancer-wisconsin-wLabels-train.csv"
inputData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\letter-recognition-wLabels-train.csv"
inputData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\ecoli-wLabels-train.csv"
inputData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\agaricus-lepiota-wLabels-train.csv"
inputData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\lp5-formatted-wLabels-train.csv"

lecData = r"D:\Users\radam\Desktop\lecData.csv"

#userDataPath = inputData2
userDataPath = lecData

#add header arguement to prevent first row from being read as labels. Enable whitespace delim line for whitespace delim datasets
df = pd.read_csv(userDataPath)
columnsNamesArr = df.columns.values
targetAttribute = columnsNamesArr[-1]










# #Finding the Entropy of Target Variable
# entropyTotal = entropyS(df, targetAttribute)
# igStump = {}

# for column in columnsNamesArr[:-1]:
#     testEnt = entropyAtt(df, column , targetAttribute)
#     igStump.update({column:infoGain(entropyTotal, testEnt)})

# stump = findWinner(igStump)
# #Get uniques of winner attribute
# stumpCol = df.loc[:, stump]
# stumpColValues = stumpCol.unique()

# print(stumpColValues)

#Need This line to create branches
#Do the calculation local to the for loop

# for value in stumpColValues:
#     dfTEST = df[df[stump]==value]
#     dfAtt = dfTEST.drop(columns=stump)
#     columnLabels = dfAtt.columns.values

    # for x in columnLabels:
    #     newSUM = entropyAtt(dfAtt, x, columnsNamesArr[-1])
    # entropyStumpNew = entropyS(dfTEST, columnsNamesArr[-1])
    # print(infoGain(entropyStumpNew, newSUM))
    # entropySUM = entropyAt,t(dfTEST, "Temperature", columnsNamesArr[-1] )


#dfTEST = df[df[stump]=='Overcast']
# entropyStump = entropyCalc(dfTEST, columnsNamesArr[-1])
# print(entropyStump)

#print(df["outlook"].value_counts())

# root: dictionary, the current pointed node/feature of the tree. It is contineously being updated.
# prev_feature_value: Any datatype (Int or Float or String etc.) depending on the datatype of the previous feature, the previous value of the pointed node/feature
# train_data: a pandas dataframe/dataset
# label: string, name of the label of the dataframe (=Play Tennis)
# class_list: list, unique classes of the label (=[Yes, No]).
# returns: None

# def make_tree(root, prev_feature_value, train_data, label, class_list):
#     if train_data.shape[0] != 0: #if dataset becomes enpty after updating
#         max_info_feature = find_most_informative_feature(train_data, label, class_list) #most informative feature
#         tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #getting tree node and updated dataset
#         next_root = None
        
#         if prev_feature_value != None: #add to intermediate node of the tree
#             root[prev_feature_value] = dict()
#             root[prev_feature_value][max_info_feature] = tree
#             next_root = root[prev_feature_value][max_info_feature]
#         else: #add to root of the tree
#             root[max_info_feature] = tree
#             next_root = root[max_info_feature]
        
#         for node, branch in list(next_root.items()): #iterating the tree node
#             if branch == "?": #if it is expandable
#                 feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
#                 make_tree(next_root, node, feature_value_data, label, class_list) #recursive call with updated dataset