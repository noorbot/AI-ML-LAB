import pandas as pd

# input .csv file names
file_name = "data/letter-recognition.csv"         # <-- change file name to match data set
output_file = "data/lab5/letter_recognition.csv"

# input target variable columns 
target_col = 0                               # <-- change to match index of target column

# read from data file and save to pandas DataFrame 'data'
data = pd.read_csv(file_name, header = None)
len_data = len(data)

# store contents of target variable column to 'store_target'
store_target = data.iloc[: , target_col:(target_col+1)].values

# first remove the target column from the data, then add it to the rightmost column
data = data.drop(columns = [target_col])
data[-1] = store_target



# store data as pandas DataFrame types and save to csv files
data.to_csv(output_file, index = False, header = False)