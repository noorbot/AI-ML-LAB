import pandas as pd
import numpy as np
import math

train_file_name = "results/letter_recognition_train_data.csv"         # <-- change file name to match data set
test_file_name = "results/letter_recognition_test_data.csv"

# read from data file and save to pandas DataFrame 'data'
train_data = pd.read_csv(train_file_name, header = None)
test_data = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe
# train_data = train_data.iloc[:100, :]
# test_data = test_data.iloc[:1, :]

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



def calculateBayes(target_var_probs, attribute_probs, test_row):

    nearest = [0]
    #bayes_array = pd.DataFrame(index=[target_classes], columns=range(num_attributes+3))
    bayes_df = pd.DataFrame(columns=['t_class','bayes'])
    # print(bayes_df)

    for i, t_class in zip(range(num_target_classes), target_classes): # compare the test instance against each train instance
        bayes = 1
        bayes = bayes * target_var_probs.iloc[i,2]

        for n_attribute in range(num_attributes): # for each attribute...
            attribute_val = test_row[n_attribute]
            # print("attribute: " + str(attribute_val)); print('t_class: ' + str(t_class))
            term = attribute_probs.loc[(attribute_probs.loc[:,'attribute'] == n_attribute) & (attribute_probs.loc[:,'a_class'] == attribute_val) & (attribute_probs.loc[:,'t_class']==t_class)]
            # print('TERM') ; print(term)
            if term.empty:
                P=0
            else:
                P = term.iat[0,3]
            # print(P)
            bayes = bayes *P

        new_row = pd.DataFrame([{'t_class' : t_class,'bayes' : bayes}])
        bayes_df = pd.concat([bayes_df, new_row], axis=0, ignore_index=True)
        
    # print(bayes_df)
    bayes_sorted = bayes_df.sort_values('bayes', ascending=False) # sort the ED_array ascending to find the smallest EDs/ nearest neighbours
    nearest = bayes_sorted.iloc[:1] # take the top row
    # print(nearest)
    return nearest



def evaluate(train_data, test_data): # method to feed in test instances and calculate algorithm accuracy
    correct_predict = 0
    wrong_predict = 0
    for instance in range(num_test_instances):  # for each test instance ...
        test_row = test_data.iloc[instance,:] # take the row of interest

        nearest = calculateBayes(target_var_probs, attribute_probs, test_row)  # feed this instance into the findNearest() method

        classification = nearest.iloc[0,0]
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

attribute_probs = calc_attribute_probs(train_data, target_var_probs)
print(attribute_probs)

accuracy = evaluate(train_data, test_data) # call the algorithm
print("\nAccuracy: " + str(accuracy))