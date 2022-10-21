import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

train_file_name = "results/wdbc_train_data.csv"         # <-- change file name to match data set
test_file_name = "results/wdbc_test_data.csv"

# read from data file and save to pandas DataFrame 'data'
train_data = pd.read_csv(train_file_name, header = None)
test_data = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe
# train_data = train_data.iloc[0:1, :]
# test_data = test_data.iloc[0:1, :]


def bin_numerical_data(train_data, test_data):
    num_bins = 20
    num_attributes = train_data.shape[1] - 1
    num_instances_train = train_data.shape[0]
    num_instances_test = test_data.shape[0]
    # determine which columns have numerical values then take range and split into x bins
    for i in range(num_attributes):
        print(is_numeric_dtype(train_data.iloc[0,i]))
        if (is_numeric_dtype(train_data.iloc[0,i])) == True:
            max = np.max(train_data.iloc[:,i])
            min = np.min(train_data.iloc[:,i])
            attribute_range = max - min
            
            for j in range(num_instances_train):
                print("\nvalue: " + str(train_data.iloc[j, i]))
                for bin in range(num_bins):
                    if train_data.iloc[j, i] >= max - (bin+1) * (attribute_range / num_bins) :
                        train_data.loc[j, i] = max - (bin+1/2)*(attribute_range / num_bins)
                        print("set equal to " + str(max - (bin+1/2)*(attribute_range / num_bins)))
                        break 

    for k in range(num_attributes):
        print(is_numeric_dtype(test_data.iloc[0,k]))
        if (is_numeric_dtype(test_data.iloc[0,k])) == True:
            max = np.max(train_data.iloc[:,k])
            min = np.min(train_data.iloc[:,k])
            attribute_range = max - min

            for l in range(num_instances_test):
                print("\nvalue: " + str(test_data.iloc[l,k]))
                for bin in range(num_bins):
                    print("tester: " + str(max - (bin+1) * (attribute_range / num_bins)))
                    print("max: " + str(max) + "   birn: " + str(bin) + "    attribute range: " + str(attribute_range) + "    num bins: " + str(num_bins))
                    if test_data.iloc[l, k] >= max - (bin+1) * (attribute_range / num_bins) :
                        test_data.loc[l, k] = max - (bin+1/2)*(attribute_range / num_bins)
                        print("set equal to " + str(max - (bin+1/2)*(attribute_range / num_bins)))
                        break 

    return train_data, test_data


train_data, test_data = bin_numerical_data(train_data, test_data)
print(train_data)
print(test_data)