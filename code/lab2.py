import pandas as pd
import numpy as np

train_file_name = "data/example.csv"         # <-- change file name to match data set
test_file_name = "data/example_test.csv"

# read from data file and save to pandas DataFrame 'data'
data = pd.read_csv(train_file_name, header = None)
test_data_m = pd.read_csv(test_file_name, header=None) #importing test dataset into dataframe
# data = pd.DataFrame([ 
# ['Sunny',   'Weak',   'No'], 
# ['Sunny',   'Strong', 'No'],
# ['Overcast','Weak',   'Yes'], 
# ['Rain',    'Weak',   'Yes'],
# ['Rain',    'Weak',   'Yes'], 
# ['Rain',    'Strong', 'No'],
# ['Overcast','Strong', 'Yes'], 
# ['Sunny',   'Weak',   'No'],
# ['Sunny',   'Weak',   'Yes'],
# ['Rain',    "Weak",   'Yes'],
# ['Sunny',   'Strong', 'Yes'],
# ['Overcast','Strong', 'Yes'],
# ['Overcast','Weak',   'Yes'],
# ['Rain',    'Strong', 'No']
# ], columns=['Outlook', 'Wind', 'Tennis'])

print(data.head())

# count the number of attributes
num_attributes = data.shape[1] - 1
num_instances = data.shape[0]
print('\nNumber of attributes: ' + str(num_attributes))
print('Number of instances: ' + str(num_instances))

target_col = -1
target_classes = data.iloc[:, -1].unique()
print('Target variable classes: ' + str(target_classes))
num_target_classes = np.count_nonzero(target_classes)

class_prop = []
attribute_class_prop = []

#target_cols = []


def calc_target_entropy(dataset):
    target_entropy = 0
    # find the entropy of the target variable (include each class)
    for clas in target_classes:                 # find proportions
        prop = dataset.iloc[:,-1].value_counts()[clas] / num_instances
        class_prop.append(prop)

    for i in range(0, num_target_classes):      #calculate entropy of target var
        target_entropy = target_entropy - class_prop[i] * np.log2(class_prop[i])

    print('Target var entropy: ' + str(target_entropy))
    return(target_entropy)

def calc_entropies(dataset):
    Attribute_Entropies = pd.DataFrame(index=np.arange(0), columns=np.arange(4 + num_target_classes))
    prev_num_attribute_classes = 0
    attr_class_entropy = 0

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

    print(Attribute_Entropies)
    return Attribute_Entropies

def calc_IGs(dataset, target_entropy, Attribute_Entropies):
    prev_num_attribute_classes = 0
    Attribute_IG = pd.DataFrame(index=np.arange(0), columns=np.arange(2))

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
            print('proportion')
            print(Attribute_Entropies.iloc[prev_num_attribute_classes + i , 2])
            #print('entropy')
            #print(Attribute_Entropies.iloc[prev_num_attribute_classes + i , -1])

        prev_num_attribute_classes = prev_num_attribute_classes + num_attribute_classes

    print(Attribute_IG)
    return Attribute_IG

def find_stump(dataset):
    target_entropy = calc_target_entropy(dataset)

    Attribute_Entropies = pd.DataFrame(index=np.arange(0), columns=np.arange(4 + num_target_classes))
    Attribute_Entropies = calc_entropies(dataset)

    Attribute_IG = pd.DataFrame(index=np.arange(0), columns=np.arange(2))
    Attribute_IG = calc_IGs(dataset, target_entropy, Attribute_Entropies)

    # pick attribute with highest information gain as stump
    stump_col = Attribute_IG[Attribute_IG.iloc[:, 1] == Attribute_IG.iloc[:, 1].max()].iloc[0,0]
    print('Stump attribute index: ' + str(stump_col))
    return(stump_col)

# def split_data(dataset):
#     # split the data by the classes fo the chosen stump
#     # then would need to check for homo (unique) to decide if its a leaf or if need to continue branching (repeat process)
#     stump_attribute_classes = dataset.iloc[:, stump_col].unique()
#     print('\nStump classes: ' + str(stump_attribute_classes))
#     num_stump_attribute_classes = np.count_nonzero(stump_attribute_classes)
#     print('Num Stump classes: ' + str(num_stump_attribute_classes))

#     data_split = {}
#     for stump_class_attribute, stump_class_num in zip(stump_attribute_classes ,range(0,num_stump_attribute_classes)):      # make a new datasets by splitting by stump attribute classes
#         data_split[stump_class_num] = dataset[dataset.iloc[:, stump_col]== stump_class_attribute].copy()
#         print(data_split[stump_class_num])
#         #print((data_split[stump_class_num].iloc[:,-1] == data_split[stump_class_num].iloc[0,-1]).all() )
#         if (data_split[stump_class_num].iloc[:,-1] == data_split[stump_class_num].iloc[0,-1]).all(): 
#             print('homogenous')
#         else: 
#             print('not homogenous')


def generate_sub_tree(stump_col, train_data):
    # the goal is to pass the stump feature in
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


def id3(data_m, target_col):
    data = data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    #class_list = data.iloc[:,target_col].unique() #getting unqiue classes of the target col
    make_tree(tree, None, data) #start calling recursion
    return tree


tree = id3(data, -1)
print(tree)

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


def evaluate(tree, test_data_m):
    correct_predict = 0
    wrong_predict = 0
    num_testdata_rows = test_data_m.shape[0]
    for row in range(0,num_testdata_rows): #for each row in the dataset
        print(row)
        result = predict(tree, test_data_m.iloc[row,:]) #predict the row
        print('result:  ' + str(result))
        print('thing:   ' + str(test_data_m.iloc[row, target_col]))
        if result == test_data_m.iloc[row, target_col]: #predicted value and expected value is same or not
            correct_predict += 1 #increase correct count
            print("YES")
        else:
            wrong_predict += 1 #increase incorrect count
            print("NO")
    accuracy = correct_predict / (correct_predict + wrong_predict) #calculating accuracy
    return accuracy



accuracy = evaluate(tree, test_data_m) #evaluating the test dataset
print(accuracy)

