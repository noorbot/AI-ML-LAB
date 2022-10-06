from xml.etree.ElementPath import find
import pandas as pd
import numpy as np

def entropyCalc(dataframe, column):
    df = dataframe
    entropy = 0
    targetCol = df.loc[:, column]
    instances = targetCol.value_counts()

    for x in instances:
        pi = x / len(targetCol)
        entropy += -pi*np.log2(pi)
    
    return entropy

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
        entropyTotal += (-1)*weightedFrac*(entropyCalc(svSamples, targetVariable))

    return entropyTotal

def infoGain (entS, entSVTot):
    return entS + entSVTot

def findWinner(list):
    attributeWin = max(list, key=list.get)
    return attributeWin

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

#Finding the Entropy of Target Variable
entropyS = entropyCalc(df, columnsNamesArr[-1])
igStump = {}

for column in columnsNamesArr[:-1]:
    testEnt = entropyAtt(df, column , columnsNamesArr[-1])
    igStump.update({column:infoGain(entropyS, testEnt)})

stump = findWinner(igStump)
#Get uniques of winner attribute
stumpCol = df.loc[:, stump]
stumpColValues = stumpCol.unique()

print(stumpColValues)

for value in stumpColValues:
    dfTEST = df[df[stump]==value]


dfTEST = df[df[stump]=='Overcast']
entropyStump = entropyCalc(dfTEST, columnsNamesArr[-1])
print(dfTEST)
print(entropyStump)