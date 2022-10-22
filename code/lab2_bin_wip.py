import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

train_file_name = "results/wdbc_train_data.csv"         # <-- change file name to match data set
test_file_name = "results/wdbc_test_data.csv"

binned_train_file = "binned/wdbc_train_data.csv"
binned_test_file = "binned/wdbc_test_data.csv"

# read from data file and save to pandas DataFrame 'data'
train_data = pd.read_csv(train_file_name, header = None)
test_data = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe
# train_data = train_data.iloc[:, :3]
# test_data = test_data.iloc[:, :3]

# train_data_udpated = pd.Dataframe()
# test_data_udpated = pd.Dataframe()

print(train_data)
print(test_data)

def bin_numerical_data(train_data, test_data):
    train_data_updated = train_data.copy()
    test_data_updated = test_data.copy()
    num_bins = 20
    num_attributes = train_data.shape[1] - 1
    num_instances_train = train_data.shape[0]
    num_instances_test = test_data.shape[0]
    #determine which columns have numerical values then take range and split into x bins
    for i in range(num_attributes):
        # print('i type: ' + str(type(i)))
        #print(is_numeric_dtype(train_data.iloc[0,i]))
        if (is_numeric_dtype(train_data.iloc[0,i])) == True:
            # print('maxerrr: ' + str(np.max(train_data.iloc[:,i])))
            max = np.max([np.max(train_data.iloc[:,i]), np.max(test_data.iloc[:,i])])
            min = np.min([np.min(train_data.iloc[:,i]), np.min(train_data.iloc[:,i])])
            attribute_range = max - min
            # print('max type: ' + str(type(max)))
            # print('\nmax: ' + str(max) + "    "  + 'attribute raneg: ' + str(attribute_range))

            for l in range(num_instances_train):
                # print("\nvalue: " + str(test_data.iloc[l,i]))
                for bin in range(num_bins):
                    # print("test bin: " + str(max - (bin+1) * (attribute_range / num_bins)))
                    #print("max: " + str(max) + "   birn: " + str(bin) + "    attribute range: " + str(attribute_range) + "    num bins: " + str(num_bins))
                    if train_data.iloc[l, i] >= max - (bin+1) * (attribute_range / num_bins) :
                        train_data_updated.loc[l, i] = round(max - (bin+1/2)*(attribute_range / num_bins), 5)
                        # print("set equal to " + str(max - (bin+1/2)*(attribute_range / num_bins)))
                        break 

    print("\n TRAIN SET")
    for k in range(num_attributes):
        # print('k type: ' + str(type(k)))
       # print(is_numeric_dtype(test_data.iloc[0,k]))
        if (is_numeric_dtype(test_data.iloc[0,k])) == True:
            # print('maxerrrk: ' + str(train_data.iloc[2,k]))
            max = np.max([np.max(train_data.iloc[:,k]), np.max(test_data.iloc[:,k])])
            min = np.min([np.min(train_data.iloc[:,k]), np.min(train_data.iloc[:,k])])
            attribute_range = max - min
            # print('max type: ' + str(type(max)))
            # print('\nmax: ' + str(max) + "    "  + 'attribute raneg: ' + str(attribute_range))

            for l in range(num_instances_test):
                # print("\nvalue: " + str(test_data.iloc[l,k]))
                for bin in range(num_bins):
                    # print("test bin: " + str(max - (bin+1) * (attribute_range / num_bins)))
                    #print("max: " + str(max) + "   birn: " + str(bin) + "    attribute range: " + str(attribute_range) + "    num bins: " + str(num_bins))
                    if test_data.iloc[l, k] >= max - (bin+1) * (attribute_range / num_bins) :
                        test_data_updated.loc[l, k] = round(max - (bin+1/2)*(attribute_range / num_bins), 5)
                        # print("set equal to " + str(max - (bin+1/2)*(attribute_range / num_bins)))
                        break             



    return train_data_updated, test_data_updated


train_data, test_data = bin_numerical_data(train_data, test_data)
print(train_data)
print(test_data)


train_data.to_csv(binned_train_file, index = False, header = False)
test_data.to_csv(binned_test_file, index = False, header = False)