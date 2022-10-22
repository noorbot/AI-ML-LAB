import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

train_file_name = "binned/wdbc_train_data.csv"         # <-- change file name to match data set
test_file_name = "binned/wdbc_test_data.csv"

# read from data file and save to pandas DataFrame 'data'
train_data = pd.read_csv(train_file_name, header = None)
test_data = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe

print(train_data.head())

# count the number of attributes
num_attributes = train_data.shape[1] - 1
num_instances = train_data.shape[0]
print('\nNumber of attributes: ' + str(num_attributes))
print('Number of instances: ' + str(num_instances))

target_col = -1
target_classes = train_data.iloc[:, -1].unique()
print('Target variable classes: ' + str(target_classes))
num_target_classes = np.count_nonzero(target_classes)

class_prop = []



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

def calc_target_entropy(dataset, num_instances, target_classes, num_target_classes):
    target_entropy = 0
    target_classes = dataset.iloc[:, -1].unique()   #needs to be updated with each recursion
    num_target_classes = np.count_nonzero(target_classes)
    num_instances = dataset.shape[0]
    # find the entropy of the target variable (include each class)
    for clas in target_classes:                 # find proportions
        prop = dataset.iloc[:,-1].value_counts()[clas] / num_instances
        class_prop.append(prop)

    for i in range(0, num_target_classes):      #calculate entropy of target var
        target_entropy = target_entropy - class_prop[i] * np.log2(class_prop[i])

    print('Target var entropy: ' + str(target_entropy))
    return(target_entropy)

def calc_entropies(dataset, target_classes, num_target_classes):
    Attribute_Entropies = pd.DataFrame(index=np.arange(0), columns=np.arange(4 + num_target_classes))
    prev_num_attribute_classes = 0
    attr_class_entropy = 0
    target_classes = dataset.iloc[:, -1].unique()   #needs to be updated with each recursion
    num_target_classes = np.count_nonzero(target_classes)

    # find entropy of target variable w.r.t. each attribute
    for attribute in range(0, num_attributes):                 # for each attribute ...
        attribute_classes = dataset.iloc[:, attribute].unique()   # find unique attribute classes
        num_attribute_classes = np.count_nonzero(attribute_classes)
        #print('\nattribute: ' + str(attribute))
        #print('num attribute classes: ' + str(num_attribute_classes))
        #print(attribute_classes)

        for clas, i in zip(attribute_classes, range(0,num_attribute_classes)):                         # for each class of an attribute ...
            num_attribute_class_for_target = dataset.iloc[:,attribute].value_counts()[clas]               # count instances of that class
            Attribute_Entropies.loc[len(Attribute_Entropies), [0, 1, 2]] = [attribute, clas, num_attribute_class_for_target]
            attr_class_entropy = 0
            
            for clas_1, j in zip(target_classes, range(0,num_target_classes) ):                      # for each target variable class
                #print("\nattribute class: " + str(clas) + "      target class: "  + str(clas_1))
                proppy = len(dataset[(dataset.iloc[:,attribute]==clas) & (dataset.iloc[:,-1]==clas_1)])
                #print('proppppuy' + str(proppy))
                #print(prev_num_attribute_classes)
                Attribute_Entropies.iloc[-1, 3 + j] = proppy
                if(proppy !=0):
                    attr_class_entropy = attr_class_entropy - proppy/num_attribute_class_for_target * np.log2(proppy/num_attribute_class_for_target)
                else: attr_class_entropy = attr_class_entropy +0
                Attribute_Entropies.iloc[-1, -1] = attr_class_entropy
                #print("HEERHER")
                #print(Attribute_Entropies)
            
        #prev_num_attribute_classes = prev_num_attribute_classes   should not be needed anymore

    print("Attribute Entropies:")
    print(Attribute_Entropies)
    return Attribute_Entropies

def calc_IGs(dataset, target_entropy, Attribute_Entropies, num_instances):
    prev_num_attribute_classes = 0
    Attribute_IG = pd.DataFrame(index=np.arange(0), columns=np.arange(2))
    num_instances = dataset.shape[0]

    # find information gain of each attribute
    for attribute in range(0, num_attributes):                 # for each attribute ...
        attribute_classes = dataset.iloc[:, attribute].unique()   # find unique attribute classes
        num_attribute_classes = np.count_nonzero(attribute_classes)
        Attribute_IG.loc[len(Attribute_IG), 0] = attribute
        IG = target_entropy

        #print('num attributes: ' + str(num_attributes) + '    num classes:' + str(num_attribute_classes))

        for i in range(0, num_attribute_classes):                 # for each target class ...
            IG = IG - (Attribute_Entropies.iloc[prev_num_attribute_classes + i , 2] / num_instances) * (Attribute_Entropies.iloc[prev_num_attribute_classes + i , -1])
            #print('IG')
            #print(IG)
            Attribute_IG.iloc[attribute, -1] = IG
            # print('proportion')
            # print(Attribute_Entropies.iloc[prev_num_attribute_classes + i , 2])
            #print('entropy')
            #print(Attribute_Entropies.iloc[prev_num_attribute_classes + i , -1])

        prev_num_attribute_classes = prev_num_attribute_classes + num_attribute_classes

    #print(Attribute_IG)
    return Attribute_IG

def find_stump(dataset):
    target_classes = dataset.iloc[:, -1].unique()   #needs to be updated with each recursion
    num_target_classes = np.count_nonzero(target_classes)
    num_instances = dataset.shape[0]
    
    target_entropy = calc_target_entropy(dataset, num_instances, target_classes, num_target_classes)

    Attribute_Entropies = pd.DataFrame(index=np.arange(0), columns=np.arange(4 + num_target_classes))
    Attribute_Entropies = calc_entropies(dataset, target_classes, num_target_classes)

    Attribute_IG = pd.DataFrame(index=np.arange(0), columns=np.arange(2))
    Attribute_IG = calc_IGs(dataset, target_entropy, Attribute_Entropies, num_instances)

    # pick attribute with highest information gain as stump
    stump_col = Attribute_IG[Attribute_IG.iloc[:, 1] == Attribute_IG.iloc[:, 1].max()].iloc[0,0]
    print('Stump attribute index: ' + str(stump_col))
    return(stump_col)


def generate_sub_tree(stump_col, train_data):
    # the goal is to pass the stump feature in
    target_classes = train_data.iloc[:, -1].unique()   #needs to be updated with each recursion
    stump_value_count_dict = train_data[stump_col].value_counts(sort=False) #dictionary of the count of unqiue feature value
    tree = {} #sub tree or node
    
    for stump_value, count in stump_value_count_dict.items():
        stump_value_data = train_data[train_data[stump_col] == stump_value] #dataset with only stump_col = stump_value
        
        assigned_to_node = False #flag for tracking stump_value is pure class or not
        for c in target_classes: #for each class
            class_count = stump_value_data[stump_value_data.iloc[:,target_col] == c].shape[0] #count of class c

            if class_count == count: #if all
                tree[stump_value] = c #adding node to the tree
                train_data = train_data[train_data[stump_col] != stump_value] #removing rows with stump_value
                assigned_to_node = True
        if not assigned_to_node: #not pure class
            tree[stump_value] = "?" #should extend the node, so the branch is marked with ?
            
    return tree, train_data


def make_tree(root, prev_feature_value, train_data):
    if train_data.shape[0] != 0: #if dataset becomes enpty after updating
        stump_col = find_stump(train_data) #most informative feature
        tree, train_data = generate_sub_tree(stump_col, train_data) #getting tree node and updated dataset
        next_root = None
        
        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][stump_col] = tree
            next_root = root[prev_feature_value][stump_col]
        else: #add to root of the tree
            root[stump_col] = tree
            next_root = root[stump_col]
        
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[stump_col] == node] #using the updated dataset
                make_tree(next_root, node, feature_value_data) #recursive call with updated dataset


def id3(train_data):
    data = train_data.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    #class_list = data.iloc[:,target_col].unique() #getting unqiue classes of the target col
    make_tree(tree, None, data) #start calling recursion
    return tree


def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None


def evaluate(tree, test_data):
    correct_predict = 0
    wrong_predict = 0
    num_testdata_rows = test_data.shape[0]
    for row in range(0,num_testdata_rows): #for each row in the dataset
        result = predict(tree, test_data.iloc[row,:]) #predict the row
        print('result:  ' + str(result))
        print('thing:   ' + str(test_data.iloc[row, target_col]))
        if result == test_data.iloc[row, target_col]: #predicted value and expected value is same or not
            correct_predict += 1 #increase correct count
        else:
            wrong_predict += 1 #increase incorrect count
    accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy
    return accuracy

# train_data, test_data = bin_numerical_data(train_data, test_data)
# print(train_data)
# print(test_data)

tree = id3(train_data)
print(tree)

accuracy = evaluate(tree, test_data) #evaluating the test dataset
print(tree)
print('ACCURACY: ' + str(accuracy))