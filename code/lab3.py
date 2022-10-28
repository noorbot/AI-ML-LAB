import pandas as pd
import numpy as np
import math

train_file_name = "results/ecoli_train_data.csv"         # <-- change file name to match data set
test_file_name = "results/ecoli_test_data.csv"
k = 8

# read from data file and save to pandas DataFrame 'data'
train_data = pd.read_csv(train_file_name, header = None)
test_data = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe
# train_data = train_data.iloc[:100, :]
# test_data = test_data.iloc[:5, :]

# count the number of attributes
num_attributes = train_data.shape[1] - 1
num_train_instances = train_data.shape[0]
num_test_instances = test_data.shape[0]
print('\nNumber of attributes: ' + str(num_attributes))
print('Number of train instances: ' + str(num_train_instances))
print('Number of test instances: ' + str(num_test_instances))

target_col = -1



def findNearest(train_data, test_row):
    # nearest = pd.DataFrame(index=np.arange(k), columns=range(2)) 
    nearest = [0]*k
    print(nearest)

    ED_array = pd.DataFrame(index=np.arange(num_train_instances), columns=range(1))

    for instance in range(num_train_instances):
        sum = 0
        for attribute in range(num_attributes):
            term = (test_row.iloc[attribute] - train_data.iloc[instance,attribute])**2
            sum = sum + term
        ED = math.sqrt(sum)
        # ED_array.loc[instance, 0] = instance
        ED_array.loc[instance, 0] = ED
        ED_array.loc[instance, 1] = train_data.iloc[instance,-1]
        # if ED < nearest[1]:
        #     nearest[0] = instance
        #     nearest[1] = ED
    print(ED_array)
    # print("MIN index: " + str(ED_array.loc[ED_array[0].idxmin()][0]))
    #nearest = [ED_array.idxmin()[0] , ED_array.iloc[0].min()]
    #print("NEAREST: " + str(nearest))
    print("SORTEDDD")
    print(ED_array.sort_values(0)[:k])
    ED_sorted = ED_array.sort_values(0)
    nearest = ED_sorted.iloc[:k]
    print("NEARSET ")
    print(nearest)
    return nearest


def evaluate(train_data, test_data):
    correct_predict = 0
    wrong_predict = 0
    for instance in range(num_test_instances):
        test_row = test_data.iloc[instance,:]

        nearest = findNearest(train_data, test_row)

        # print(nearest)
        # classification = train_data.iloc[nearest[0], target_col]
        classification = nearest.iloc[:,1].value_counts().idxmax()
        true_value = test_row.iloc[-1]
        print("\nclassification: " +str(classification))
        print("true value: " + str(true_value))

        if classification == true_value: #predicted value and expected value is same or not
            correct_predict += 1 #increase correct count
        else:
            wrong_predict += 1 #increase incorrect count
    accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy
    return accuracy



# nearest = findNearest(train_data, test_data)

accuracy = evaluate(train_data, test_data)
print("\nAccuracy: " + str(accuracy))







