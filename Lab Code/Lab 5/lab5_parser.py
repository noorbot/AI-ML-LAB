import pandas as pd

#Comma Delim
inputData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Breast Cancer Wisconsin (Diagnostic) Data Set\wbdc.data"
inputData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Letter Recognition Data Set\letter-recognition.data"
inputData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Mushroom Data Set\agaricus-lepiota.data"
inputData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Robot Execution Failures Data Set\lp5-formatted.csv"

#Whitespace delim
inputData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Ecoli Data Set\ecoli.data"

userDataPath = inputData1

#add header arguement to prevent first row from being read as labels. Enable whitespace delim line for whitespace delim datasets
if userDataPath == inputData1 or userDataPath == inputData3 or userDataPath == inputData4  or userDataPath == inputData5:
    df = pd.read_csv(userDataPath, header=None)
elif userDataPath == inputData2:
    df = pd.read_csv(userDataPath,delim_whitespace= True, header=None)

#Get number of columns and add a prefix for user to identify correct label to shift to the last column
labels = []
numColumns = df.shape[1]

#Ask user for column names
labelColumns = input("Do you want to label the attributes? " + "(Total Columns: " + str(numColumns) + ")\n")
if labelColumns == 'y' or labelColumns == 'Y' or labelColumns == 'Yes' or labelColumns == 'yes':
    # for x in range(numColumns):
    #     val = input("Name of Column " + str(x) + ":\n")
    #     labels.append(val)
    # df.columns = labels
    if userDataPath == inputData1:
        labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','Class']
        df.columns = labels
    elif userDataPath == inputData3:
        labels = ['x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx','lettr']
        df.columns = labels
    elif userDataPath == inputData2:
        labels = ['Sequence Name','mcg','gvh','lip','chg','aac','alm1','alm2','class']
        df.columns = labels
    elif userDataPath == inputData4:
        labels = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat','class']
        df.columns = labels
    elif userDataPath == inputData5:
        labels = ['Fx1','Fy1','Fz1','Tx1','Ty1','Tz1','Fx2','Fy2','Fz2','Tx2','Ty2','Tz2','Fx3','Fy3','Fz3','Tx3','Ty3','Tz3','Fx4','Fy4','Fz4','Tx4','Ty4','Tz4','Fx5','Fy5','Fz5','Tx5','Ty5','Tz5',Fx6,Fy6,Fz6,Tx6,Ty6,Tz6,Fx7,Fy7,Fz7,Tx7,Ty7,Tz7,Fx8,Fy8,Fz8,Tx8,Ty8,Tz8,Fx9,Fy9,Fz9,Tx9,Ty9,Tz9,Fx10,Fy10,Fz10,Tx10,Ty10,Tz10,Fx11,Fy11,Fz11,Tx11,Ty11,Tz11,Fx12,Fy12,Fz12,Tx12,Ty12,Tz12,Fx13,Fy13,Fz13,Tx13,Ty13,Tz13,Fx14,Fy14,Fz14,Tx14,Ty14,Tz14,Fx15,Fy15,Fz15,Tx15,Ty15,Tz15,Class]
        df.columns = labels

elif labelColumns == 'n' or labelColumns == 'N':
    df = df.add_prefix('X')

#Speical Case for robot data set to quickly name the 91 columns
elif labelColumns == "robotFailures" or labelColumns == "robotfailures" or labelColumns == "robot":
    robotNames = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\OG Datasets\Robot Execution Failures Data Set\robotName.csv"
    df2 = pd.read_csv(robotNames, header=None, index_col=False)
    df.columns = df2.iloc[0]

print(df)

#Ask User to verify if target varible is in the last column
shiftColumn = input("Is the target variable in the last column? \n")

#Shift target column to last position of dataframe
if shiftColumn == 'n' or shiftColumn == 'N' or shiftColumn == 'No' or shiftColumn == 'no':
    targetColumn = input("Enter label of column to shift: \n")
    tempCols = df.pop(targetColumn)
    df.insert(numColumns - 1, targetColumn, tempCols)


#Save processed dataframe
exportName = userDataPath.split(".", 1)[0]

exportLabels = input("Do you want the labels exported as well? \n")
if exportLabels == 'y' or exportLabels == 'Y' or exportLabels == 'Yes' or exportLabels == 'yes':
    df.to_csv(exportName + "-wLabels-train.csv", header=True, index = False)
elif exportLabels == 'n' or exportLabels == 'N' or exportLabels == 'No' or exportLabels == 'no':
    df.to_csv(exportName + "-train.csv", header=False, index = False)



