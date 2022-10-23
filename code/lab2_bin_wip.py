import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

train_file_name = "results/lp5_train_data.csv"         # <-- change file name to match data set
test_file_name = "results/lp5_test_data.csv"

binned_train_file = "binned/lp5_train_data.csv"
binned_test_file = "binned/lp5_test_data.csv"

# read from data file and save to pandas DataFrame 'data'
train_data = pd.read_csv(train_file_name, header = None)
test_data = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe
# train_data = train_data.iloc[:, :3]
# test_data = test_data.iloc[:, :3]


print(train_data)
print(test_data)
print(train_data.dtypes)
print(test_data.dtypes)

def bin_numerical_data(train_data, test_data): # function to split continous numerical data into bins
    train_data_updated = train_data.copy() # make copies of train and test dataframe
    test_data_updated = test_data.copy()
    num_bins = 20 # number of bins to split data into
    num_attributes = train_data.shape[1] - 1 # number of attributes in dataset
    num_instances_train = train_data.shape[0] #number of instances and train and test sets
    num_instances_test = test_data.shape[0]

    for i in range(num_attributes): # for each colummn...
        print(is_numeric_dtype(train_data.iloc[0,i]))
        if (is_numeric_dtype(train_data.iloc[0,i])) == True:
            max = np.max([np.max(train_data.iloc[:,i]), np.max(test_data.iloc[:,i])]) # find max value
            min = np.min([np.min(train_data.iloc[:,i]), np.min(train_data.iloc[:,i])]) # find min value
            attribute_range = max - min # calculate range

            for l in range(num_instances_train): # for each instance, put the value into a bin
                for bin in range(num_bins): # check through all bins
                    if train_data.iloc[l, i] >= max - (bin+1) * (attribute_range / num_bins) : # check if its the correct bin
                        train_data_updated.loc[l, i] = round(max - (bin+1/2)*(attribute_range / num_bins), 5) # update value to be middle of bin
                        break 

    for k in range(num_attributes): # reapeat process for test data set
        if (is_numeric_dtype(test_data.iloc[0,k])) == True:
            max = np.max([np.max(train_data.iloc[:,k]), np.max(test_data.iloc[:,k])]) # find max value
            min = np.min([np.min(train_data.iloc[:,k]), np.min(train_data.iloc[:,k])]) # find min value
            attribute_range = max - min  # calculate range

            for l in range(num_instances_test):
                for bin in range(num_bins):
                    if test_data.iloc[l, k] >= max - (bin+1) * (attribute_range / num_bins) : # check if its the correct bin
                        test_data_updated.loc[l, k] = round(max - (bin+1/2)*(attribute_range / num_bins), 5) # update value to be middle of bin
                        break             

    return train_data_updated, test_data_updated # return updated datasets


train_data, test_data = bin_numerical_data(train_data, test_data)    # call function
print(train_data)
print(test_data)


train_data.to_csv(binned_train_file, index = False, header = False)  # save results to .csv files
test_data.to_csv(binned_test_file, index = False, header = False)