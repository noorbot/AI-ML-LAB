import pandas as pd
import numpy as np

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

#Test Calculation For Entropy
#Finding the Entropy of Target Variable

entropyTV = 0
targetCol = df.iloc[: , -1]
print(targetCol.unique())
instances = targetCol.value_counts()

for x in instances:
    pi = x / len(targetCol)
    entropyTV += -pi*np.log2(pi)

print(entropyTV)