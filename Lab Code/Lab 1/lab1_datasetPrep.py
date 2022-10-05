import pandas as pd

#Comma Delim
inputData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Breast Cancer Wisconsin (Diagnostic) Data Set\breast-cancer-wisconsin.data"
inputData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Letter Recognition Data Set\letter-recognition.data"
inputData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Mushroom Data Set\agaricus-lepiota.data"
inputData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Robot Execution Failures Data Set\lp1-formatted.csv"

#Whitespace delim
inputData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Ecoli Data Set\ecoli.data"

userDataPath = inputData3

#add header arguement to prevent first row from being read as labels. Enable whitespace delim line for whitespace delim datasets
#df = pd.read_csv(userDataPath,delim_whitespace= True, header=None)
df = pd.read_csv(userDataPath, header=None)

#Get number of columns and add a prefix for user to identify correct label to shift to the last column
labels = []
numColumns = df.shape[1]

#Ask user for column names
labelColumns = input("Do you want to label the attributes? " + "(Total Columns: " + str(numColumns) + ")\n")
if labelColumns == 'y' or labelColumns == 'Y':
    for x in range(numColumns):
        val = input("Name of Column " + str(x) + ":\n")
        labels.append(val)
    df.columns = labels
elif labelColumns == 'n' or labelColumns == 'N':
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
exportName = userDataPath.split(".", 1)[0]

exportLabels = input("Do you want the labels exported as well? \n")
if exportLabels == 'y' or exportLabels == 'Y':
    trainSet.to_csv(exportName + "-wLabels-train.csv", header=True, index = False)
    testSet.to_csv(exportName + "-wLabels-test.csv", header=True, index = False)
elif exportLabels == 'n' or exportLabels == 'N':
    trainSet.to_csv(exportName + "-train.csv", header=False, index = False)
    testSet.to_csv(exportName + "-test.csv", header=False, index = False)



