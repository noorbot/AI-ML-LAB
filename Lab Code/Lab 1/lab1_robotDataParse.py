import pandas as pd

#Get path to dataset file
inputData = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\ENGR 3150U Data Sets\Robot Execution Failures Data Set\lp1.csv"

#Create inital variables for parser
counter = 0
ser = pd.Series(dtype=object)
rowPrev = ser

#Create function to convert list element into a series and drop empty cells
def conList(x):
    series = pd.Series(x)
    return series.dropna()

#add header arguement to prevent first row from being read as labels
df = pd.read_csv(inputData, header=None)

#Creates a List of Dictionaries from created DataFrame
df_dict = df.to_dict('records')
df2 = pd.DataFrame()

#Iterate through dictionary and concatenate each observation window into one row and add to a new dataframe
for x in df_dict:
    if counter <= 16:
        rowTest = conList(x)
        rowNew = pd.concat([rowPrev,rowTest], ignore_index=True)
        rowPrev = rowNew
        counter+=1
    elif counter > 16 and counter != 17:
        counter+=1
    else:
        df2 = pd.concat([df2, rowNew.to_frame().T])
        counter = 0
        rowNew = pd.Series(dtype=object)
        rowPrev = pd.Series(dtype=object)

#Print new dataframe to verify funtionality
print(df2)

#Export reformated data
df2.to_csv(r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\ENGR 3150U Data Sets\Robot Execution Failures Data Set\lp1-formatted.csv", header=False, index = False)
