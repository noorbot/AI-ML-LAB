import pandas as pd
import numpy as np
import math

train_file_name = "results/letter_recognition_train_data.csv"         # <-- change file name to match data set
test_file_name = "results/letter_recognition_test_data.csv"
k = 1

# read from data file and save to pandas DataFrame 'data'
train_data = pd.read_csv(train_file_name, header = None)
test_data = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe

# count the number of attributes
num_attributes = train_data.shape[1] - 1
num_train_instances = train_data.shape[0]
num_test_instances = test_data.shape[0]
print('\nNumber of attributes: ' + str(num_attributes))
print('Number of train instances: ' + str(num_train_instances))
print('Number of test instances: ' + str(num_test_instances))

target_col = -1
# target_classes = train_data.iloc[:, -1].unique()
# print('Target variable classes: ' + str(target_classes))
# num_target_classes = np.count_nonzero(target_classes)

def ED():  # function to calculate Euclidean Distance
    print('hi')


def findNearest(train_data, test_data):
    print('o')
    nearest = [0, 1000000]
    for instance in range(num_train_instances):
        sum = 0
        for attribute in range(num_attributes):
            term = (test_data.iloc[0,attribute] - train_data.iloc[instance,attribute])**2
            sum = sum + term
        ED = math.sqrt(sum)
        if ED < nearest[1]:
            nearest[0] = instance
            nearest[1] = ED
    return nearest
        

nearest = findNearest(train_data, test_data)
print(nearest)
classification = train_data.iloc[nearest[0], target_col]
print("classification: " +str(classification))
print("true value: " + str(test_data.iloc[0,-1]))






