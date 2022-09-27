import pandas as pd
import numpy as np

file_name = "lp1good.csv"         # <-- change file name to match data set
train_file = "train_data.csv"
test_file = "test_data.csv"

# input target variable columns and test data split
target_col = 0                               # <-- change to match index of target column
test_data_split = 0.2

# read from data file and save to pandas DataFrame 'data'
data = pd.read_csv(file_name, header = None)
len_data = len(data)
print(data.head())
print(data.shape)

data[7] = np.nan


for i in range(1, 5):
    for j in range(1,7):
        data.iloc[0,j] = data.iloc[i,j]

print(data.head())
