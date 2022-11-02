import pandas as pd
import numpy as np


train_file_name = "results/mushroom_train_data.csv"         # <-- change file name to match data set
test_file_name = "results/mushroom_test_data.csv"
k = 1 # <-- set k value

# read from data file and save to pandas DataFrame 'data'
train_data = pd.read_csv(train_file_name, header = None)
test_data = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe
# train_data = train_data.iloc[:100, :]
test_data = test_data.iloc[:800, :]

# count the number of attributes
num_attributes = train_data.shape[1] - 1
num_train_instances = train_data.shape[0]
num_test_instances = test_data.shape[0]
print('\nNumber of attributes: ' + str(num_attributes))
print('Number of train instances: ' + str(num_train_instances))
print('Number of test instances: ' + str(num_test_instances))

target_col = -1 # set rightmost column to to target variable column


# Since the mushroom dataset has categorical attributes, it is not possible to find euclidean distance
# Instead, we need to see which training case matches the test sample best
def findNearest(train_data, test_row): # method that find the k nearest neighbours

    nearest = [0]*k
    match_array = pd.DataFrame(index=np.arange(num_train_instances), columns=range(1))

    for instance in range(num_train_instances): # compare the test instance against each train instance
        matches = 0
        for attribute in range(num_attributes): # for each attribute...
            if(train_data.iloc[instance, attribute]== test_row.iloc[attribute]): # if the attributes match, add 1 to variables matches
                matches = matches + 1
            
        match_array.loc[instance, 0] = matches # save number of matches for each test instance into an array
        match_array.loc[instance, 1] = train_data.iloc[instance,-1]
    match_sorted = match_array.sort_values(0, ascending=False) # sort the matches in descneding order
    nearest = match_sorted.iloc[:k] # take the k highest matches and save to an array
    return nearest


def evaluate(train_data, test_data): # method to feed in test instances and calculate algorithm accuracy
    correct_predict = 0
    wrong_predict = 0
    for instance in range(num_test_instances):  # for each test instance ...
        test_row = test_data.iloc[instance,:] # take the row of interest

        nearest = findNearest(train_data, test_row)  # feed this instance into the findNearest() method

        classification = nearest.iloc[:,1].value_counts().idxmax() # clasify it by using the target variable class that is most common in the k nearest
        true_value = test_row.iloc[-1]
        print("\nclassification: " +str(classification))
        print("true value: " + str(true_value))

        if classification == true_value: #predicted value and expected value is same or not
            correct_predict += 1 #increase correct count
        else:
            wrong_predict += 1 #increase incorrect count
        current_accuracy =  accuracy = correct_predict / (correct_predict + wrong_predict)
        print("Accuracy at test #" + str(instance) + ": " + str(current_accuracy))
    accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy
    return accuracy



accuracy = evaluate(train_data, test_data) # call the algorithm
print("\nAccuracy: " + str(accuracy))


# Tim!! What time are you heading over to Matt's?
# ok sick i do not