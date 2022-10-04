import pandas as pd

#Comma Delim
inputData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\ENGR 3150U Data Sets\Breast Cancer Wisconsin (Diagnostic) Data Set\breast-cancer-wisconsin.data"
inputData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\ENGR 3150U Data Sets\Letter Recognition Data Set\letter-recognition.data"
inputData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\ENGR 3150U Data Sets\Mushroom Data Set\agaricus-lepiota.data"
inputData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\ENGR 3150U Data Sets\Robot Execution Failures Data Set\lp1-formatted.csv"

#Whitespace delim
inputData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\ENGR 3150U Data Sets\Ecoli Data Set\ecoli.data"

userDataPath = inputData1

#add header arguement to prevent first row from being read as labels. Enable whitespace delim line for whitespace delim datasets
#df = pd.read_csv(userDataPath,delim_whitespace= True, header=None)
df = pd.read_csv(userDataPath, header=None)

#Get number of columns and add a prefix for user to identify correct label to shift to the last column
numColumns = df.shape[1]
df = df.add_prefix('X')
print(df)

#Ask User to verify if target varible is in the last column
shiftColumn = input("Is the target variable in the last column? \n")

#Shift target column to last position of dataframe
if shiftColumn == 'n' or shiftColumn == 'N':
    targetColumn = input("Enter label of column to shift: \n")
    tempCols = df.pop(targetColumn)
    df.insert(numColumns - 1, targetColumn, tempCols)

#Split Data according to user desired split
userSplit = float(input("What Percentage of the data should be test data? \n"))
testSet = df.sample(frac = userSplit/100)
testDropSet = df.drop(testSet.index)

trainSet = testDropSet.sample(frac = 1)

#Save processed dataframe
trainSet.to_csv(r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\ENGR 3150U Data Sets\Breast Cancer Wisconsin (Diagnostic) Data Set\train-breast-cancer-wisconsin.csv", header=False, index = False)
testSet.to_csv(r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\ENGR 3150U Data Sets\Breast Cancer Wisconsin (Diagnostic) Data Set\test-breast-cancer-wisconsin.csv", header=False, index = False)



