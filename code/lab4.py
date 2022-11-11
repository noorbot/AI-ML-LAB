import pandas as pd
import numpy as np

train_file_name = "binned/ecoli_train_data.csv"     # <-- change file names to match data set (use binned data for numerical datasets)
test_file_name = "binned/ecoli_test_data.csv"

train_data = pd.read_csv(train_file_name, header = None)          # read from data file and save to pandas DataFrames
test_data = pd.read_csv(test_file_name, header=None)

# get important data features
num_attributes = train_data.shape[1] - 1                          # count of data attributes
num_train_instances = train_data.shape[0]                         # count of train instances
num_test_instances = test_data.shape[0]                           # count of test instances
target_classes = train_data.iloc[:, -1].unique()                  # identify the target variables classes
num_target_classes = np.count_nonzero(target_classes)             # count the unique target variable classes
print('\nNumber of attributes: ' + str(num_attributes))
print('Number of train instances: ' + str(num_train_instances))
print('Number of test instances: ' + str(num_test_instances))
print('target class names: ' + str(target_classes))

target_col = -1 # set rightmost column to to target variable column


def calc_target_var_probs(train_data):   # function to calculate all P(h) for Bayes
    target_var_probs = pd.DataFrame(columns=['target class', 'count', 'P'])   # empty pandas dataframe created for storing the P(h) data

    for t_class in target_classes:       # for each target class...
        count = train_data.iloc[:,-1].value_counts()[t_class] # count how many times this target class appears in the train dataset
        prop = count / num_train_instances   # calculate proportion of all train instances that are this target class
        new_row = pd.DataFrame([{'target class' : t_class, 'count' : count, 'P' : prop}])   # save data row into a new dataframe
        target_var_probs = pd.concat([target_var_probs, new_row], axis=0, ignore_index=True) # add this row to the dataframe for storing P(h)
    return target_var_probs


def calc_attribute_probs(train_data, target_var_probs):   # function to calculate all P(D|h) for Bayes
    attribute_probs = pd.DataFrame(columns=['attribute', 'a_class', 't_class', 'P'])   # empty pandas dataframe created for storing the P(D|h) data

    for n_attribute in range(num_attributes):  # for each attribute...
        attribute_classes = train_data.iloc[:, n_attribute].unique() # identify the attribute classes (columns)

        for a_class in attribute_classes:  # for each attribute class in that column...

            for j, t_class in zip(range(num_target_classes), target_classes):  # for each possible target variable class... (hypothesis)
                count = len(train_data[(train_data.iloc[:,n_attribute]==a_class) & (train_data.iloc[:,-1]==t_class)])   # count number of times this attribute class appears with this target variable
                prop = count / target_var_probs.iloc[j,1]  # calculate P(D|h) by dividing the count by the number of times this target variable appears in the training data
                new_row = pd.DataFrame([{'attribute': n_attribute, 'a_class' : a_class, 't_class' : t_class, 'P' : prop}])      # save data row into a new dataframe
                attribute_probs = pd.concat([attribute_probs, new_row], axis=0, ignore_index=True)    # add this row to the dataframe for storing P(D|h)
    return(attribute_probs)


def calculateBayes(target_var_probs, attribute_probs, test_row):   # function to perform Bayes calculation max[P(h|D)] = max[P(D|h)*P(h)]
    maxPhD = [0]            # initialize empty array for storing max[P(h|D)]
    bayes_df = pd.DataFrame(columns=['t_class','bayes'])     # initialize empty dataframe for storing all P(h|D)

    for i, t_class in zip(range(num_target_classes), target_classes): # for each target variable class, compare the test instance against each train instance
        bayes = target_var_probs.iloc[i,2]  # first set bayes equal to P(h) 

        for n_attribute in range(num_attributes): # for each attribute of the test row...
            attribute_val = test_row[n_attribute] # find attribute of interest
            # print("attribute: " + str(attribute_val)); print('t_class: ' + str(t_class))
            term = attribute_probs.loc[(attribute_probs.loc[:,'attribute'] == n_attribute) & (attribute_probs.loc[:,'a_class'] == attribute_val) & (attribute_probs.loc[:,'t_class']==t_class)] # find P(D|h) value with corresponding attribute, attribute class, and target variable class in the attribute_probs dataframe 
            # print('TERM') ; print(term)
            if term.empty:  # if no corresponding P(D|h) value is found (can happen if this combination of attribute class and target class appears in the train data)
                P=0 # set P(h|D) to zeo
            else: 
                P = term.iat[0,3] # set P(h|D) using correct column 'P' 
            # print(P)
            bayes = bayes *P # multiply P (h|D) to bayes

        new_row = pd.DataFrame([{'t_class' : t_class,'bayes' : bayes}]) # save this bayes value to a new row
        bayes_df = pd.concat([bayes_df, new_row], axis=0, ignore_index=True) # add this row to the bayes_df dataframe
        
    # print(bayes_df)
    bayes_sorted = bayes_df.sort_values('bayes', ascending=False) # sort the bayes_df by descending bayes values to find the max P(h|D)
    maxPhD = bayes_sorted.iloc[:1] # take the top row (max)
    print(maxPhD)
    return maxPhD



def evaluate(test_data): # method to feed in test instances and calculate algorithm accuracy
    correct_predict = 0
    wrong_predict = 0
    for instance in range(num_test_instances):  # for each test instance ...
        test_row = test_data.iloc[instance,:] # take the row of interest

        maxPhD = calculateBayes(target_var_probs, attribute_probs, test_row)  # feed this instance into the calculateBayes() method

        classification = maxPhD.iloc[0,0]  # take classificatioin as output of calculateBayes() function - take the target class
        true_value = test_row.iloc[-1]   # take the true target class
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


target_var_probs = calc_target_var_probs(train_data)    # call function to calculate all possible P(h) so that this only needs to be done once
print(target_var_probs)

attribute_probs = calc_attribute_probs(train_data, target_var_probs)   # call function to calculate all possible P(D|h) so that this only needs to be done once
print(attribute_probs)

accuracy = evaluate(train_data, test_data)   # call the algorithm and evaluate its accuracy
print("\nAccuracy: " + str(accuracy))