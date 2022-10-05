import pandas as pd

inputData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Letter Recognition Data Set\letterRecog-train.csv"

userDataPath = inputData2

#add header arguement to prevent first row from being read as labels. Enable whitespace delim line for whitespace delim datasets
#df = pd.read_csv(userDataPath,delim_whitespace= True, header=None)
df = pd.read_csv(userDataPath, header=None)

#Test Calculation For Entropy
print(df)