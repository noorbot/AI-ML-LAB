import pandas as pd
import numpy as np

def entropyCalc(dataframe, column):
    # columnsNamesArr = df.columns.values
    df = dataframe
    entropy = 0

    # targetCol = df.loc[: , columnsNamesArr[-1]]
    targetCol = df.loc[:, column]
    instances = targetCol.value_counts()

    for x in instances:
        pi = x / len(targetCol)
        entropy += -pi*np.log2(pi)
    
    return entropy


def entropyAtt(df, attribute, targetVariable):
    targetCol = df.loc[: , targetVariable]
    targetColValues = targetCol.unique()
    attCol = df.loc[:, attribute]
    attColValues = attCol.unique()

    # df2 = df[df[attribute]=="Weak"][df[targetVariable]=="No"]
    # print(df2[attribute])

    #Need To Calculate Entropy of An Atrribute with the collection of samples that match target variable
    for attVal in attColValues:
        print("For " + attVal + "Samples: \n")

        subdf = df[[attribute, targetVariable]]
        svSamples = subdf[subdf[attribute]==attVal]

        print(entropyCalc(svSamples, targetVariable))


        #for targetVal in targetColValues:
            # svSamples = df[df[attribute]==attVal][df[targetVariable]==targetVal]
            # print(svSamples[attribute])
            # print(len(svSamples))
            # sv = df[df[attribute]==attVal]
            # print(len(sv))
            # print(sv)
            # # print(entropyCalc(columnsNamesArr[-1]))


            








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
#Test Calculation For Entropy
#Finding the Entropy of Target Variable

entropyTV = 0
# targetCol = df.iloc[: , -1]
# print(targetCol.unique())
# instances = targetCol.value_counts()

# for x in instances:
#     pi = x / len(targetCol)
#     entropyTV += -pi*np.log2(pi)

targetCol = df.loc[: , columnsNamesArr[-1]]
instances = targetCol.value_counts()

for x in instances:
    pi = x / len(targetCol)
    entropyTV += -pi*np.log2(pi)
print(entropyTV)

# for column in range(df.shape[1] -1):
#     colTest = df.iloc[:,column]
#     print(colTest)

#for column in columnsNamesArr[:-1]:

entropyAtt(df, "Humidity", columnsNamesArr[-1])
