import pandas as pd
import numpy as np

file_name = "lp5.csv"         # <-- change file name to match data set
parsed_file = "parsed_data.csv"

# read from data file and save to pandas DataFrame 'data'
data = pd.read_csv(file_name, header = None)

data = data.dropna(how = 'all') # removes all rows that are completely empty

for new_col in range(7,91): # add columns for new parsed format
    data[new_col] = np.nan

# perform parsing 
for k in range(0, 164): # go through all 164 instances
    for i in range(1,16):  # go through all 15 measurements
        for j in range(1,7): # go through the sensor measurements
            data.iloc[k,j+(i-1)*6] = data.iloc[i+k,j]
    data.drop(labels = range(1+18*k,16+18*k), inplace=True)

# save data to output csv file 'parsed_file'
data.to_csv(parsed_file, index = False, header = False)