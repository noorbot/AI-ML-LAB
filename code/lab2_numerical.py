import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

train_file_name = "results/ecoli_train_data.csv"         # <-- change file name to match data set
test_file_name = "results/ecoli_test_data.csv"

# read from data file and save to pandas DataFrame 'data'
data = pd.read_csv(train_file_name, header = None)
test_data_m = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe

num_bins = 20
num_attributes = data.shape[1] - 1
num_instances = data.shape[0]

# determine which columns have numerical values then take range and split into x bins
for i in range(num_attributes):
    if (is_numeric_dtype(data.iloc[0,i])) == True:
        max = np.max(data.iloc[:,i])
        min = np.min(data.iloc[:,i])
        attribute_range = max - min
        
        for j in range(num_instances):
            #print("\nvalue: " + str(data.iloc[j, i]))
            for bin in range(num_bins):
                if data.iloc[j, i] >= max - (bin+1) * (attribute_range / num_bins) :
                    data.loc[j, i] = max - (bin+1/2)*(attribute_range / num_bins)
                    #print("set equal to " + str(max - (bin+1/2)*(attribute_range / num_bins)))
                    break    

print(data.head(20))

