import pandas as pd
import numpy as np
import math

train_file_name = "data/example.csv"         # <-- change file name to match data set
test_file_name = "data/example_test.csv"

# read from data file and save to pandas DataFrame 'data'
train_data = pd.read_csv(train_file_name, header = None)
test_data = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe
# train_data = train_data.iloc[:100, :]
# test_data = test_data.iloc[:800, :]

# count the number of attributes
num_attributes = train_data.shape[1] - 1
num_train_instances = train_data.shape[0]
num_test_instances = test_data.shape[0]
print('\nNumber of attributes: ' + str(num_attributes))
print('Number of train instances: ' + str(num_train_instances))
print('Number of test instances: ' + str(num_test_instances))

target_col = -1 # set rightmost column to to target variable column

target_classes = train_data.iloc[:, -1].unique()
print('target class names: ' + str(target_classes))
num_target_classes = np.count_nonzero(target_classes)

    

def calc_target_var_probs(train_data):
    #target_var_probs = pd.DataFrame(index=np.arange(num_target_classes), columns=range(2))
    target_var_probs = pd.DataFrame(columns=['target class', 'count', 'P'])

    for i, t_class in zip(range(num_target_classes), target_classes):
        # target_var_probs.loc[i, 0] = t_class # save name of target class to first column
        # prop = train_data.iloc[:,-1].value_counts()[t_class] / num_train_instances
        # target_var_probs.loc[i, 1] = prop  # save probability of this target class in second column

        count = train_data.iloc[:,-1].value_counts()[t_class]
        prop = count / num_train_instances
        new_row = pd.DataFrame([{'target class' : t_class, 'count' : count, 'P' : prop}])
        target_var_probs = pd.concat([target_var_probs, new_row], axis=0, ignore_index=True)
    return target_var_probs


def calc_attribute_probs(train_data, target_var_probs):
    attribute_probs = pd.DataFrame(columns=['attribute', 'a_class', 't_class', 'P'])

    for n_attribute in range(num_attributes):
        attribute_classes = train_data.iloc[:, n_attribute].unique()
        num_attribute_classes = np.count_nonzero(attribute_classes)

        for i, a_class in zip(range(num_attribute_classes), attribute_classes):

            for j, t_class in zip(range(num_target_classes), target_classes):
                count = len(train_data[(train_data.iloc[:,n_attribute]==a_class) & (train_data.iloc[:,-1]==t_class)])
                prop = count / target_var_probs.iloc[j,1]
                new_row = pd.DataFrame([{'attribute': n_attribute, 'a_class' : a_class, 't_class' : t_class, 'P' : prop}])
                attribute_probs = pd.concat([attribute_probs, new_row], axis=0, ignore_index=True)

    return(attribute_probs)



def calculateBayes(train_data, test_row):

    nearest = [0]*k
    ED_array = pd.DataFrame(index=np.arange(num_train_instances), columns=range(1))

    for instance in range(num_train_instances): # compare the test instance against each train instance
        sum = 0
        for attribute in range(num_attributes): # for each attribute...
            term = (test_row.iloc[attribute] - train_data.iloc[instance,attribute])**2 # calculation of Euclidean distance (ED)
            sum = sum + term
        ED = math.sqrt(sum)
        ED_array.loc[instance, 0] = ED # use ED_array to keep track of ED for each test instance
        ED_array.loc[instance, 1] = train_data.iloc[instance,-1]
    ED_sorted = ED_array.sort_values(0) # sort the ED_array ascending to find the smallest EDs/ nearest neighbours
    nearest = ED_sorted.iloc[:k] # take the k nearest
    return nearest



def evaluate(train_data, test_data): # method to feed in test instances and calculate algorithm accuracy
    correct_predict = 0
    wrong_predict = 0
    for instance in range(num_test_instances):  # for each test instance ...
        test_row = test_data.iloc[instance,:] # take the row of interest

        nearest = calculateBayes(train_data, test_row)  # feed this instance into the findNearest() method

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


target_var_probs = calc_target_var_probs(train_data)
print(target_var_probs)

attribute_array = calc_attribute_probs(train_data, target_var_probs)
print(attribute_array)

# accuracy = evaluate(train_data, test_data) # call the algorithm
# print("\nAccuracy: " + str(accuracy))