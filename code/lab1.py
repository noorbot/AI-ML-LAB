import pandas as pd

# input .csv file names
file_name = "data/wdbc.csv"         # <-- change file name to match data set
train_file = "results/wdbc_train_data.csv"
test_file = "results/wdbc_test_data.csv"

# input target variable columns and test data split
target_col = 1                               # <-- change to match index of target column
test_data_split = 0.2

# read from data file and save to pandas DataFrame 'data'
data = pd.read_csv(file_name, header = None)
len_data = len(data)

# store contents of target variable column to 'store_target'
store_target = data.iloc[: , target_col:(target_col+1)].values

# first remove the target column from the data, then add it to the rightmost column
data = data.drop(columns = [target_col])
data[-1] = store_target

# split 'train_data' and 'test_data' at random, using the ratio indicated by 'test_data_split' variable
rand_data = data.sample(frac = 1.0)
test_data = rand_data.iloc[0 : int(test_data_split * len_data)]
train_data = rand_data.iloc[int(test_data_split * len_data) : len_data]

# store data as pandas DataFrame types and save to csv files
train_data.to_csv(train_file, index = False, header = False)
test_data.to_csv(test_file, index = False, header = False)
print("Train data shape: "); print(train_data.shape)
print("Test data shape: "); print(test_data.shape)