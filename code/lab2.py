import pandas as pd
import numpy as np

file_name = "results/letter_recognition_train_data.csv"         # <-- change file name to match data set

# read from data file and save to pandas DataFrame 'data'
#data = pd.read_csv(file_name, header = None)
data = pd.DataFrame([ 
['Sunny',   'Weak',   'No'], 
['Sunny',   'Strong', 'No'],
['Overcast','Weak',   'Yes'], 
['Rain',    'Weak',   'Yes'],
['Rain',    'Weak',   'Yes'], 
['Rain',    'Strong', 'No'],
['Overcast','Strong', 'Yes'], 
['Sunny',   'Weak',   'No'],
['Sunny',   'Weak',   'Yes'],
['Rain',    "Weak",   'Yes'],
['Sunny',   'Strong', 'Yes'],
['Overcast','Strong', 'Yes'],
['Overcast','Weak',   'Yes'],
['Rain',    'Strong', 'No']
], columns=['Outlook', 'Wind', 'Tennis'])

print(data.head())

# count the number of attributes
num_attributes = data.shape[1] - 1
num_instances = data.shape[0]
print('\nNumber of attributes: ' + str(num_attributes))
print('Number of instances: ' + str(num_instances))

target_classes = data.iloc[:, -1].unique()
print('Target variable classes: ' + str(target_classes))
num_target_classes = np.count_nonzero(target_classes)

class_prop = []
attribute_class_prop = []
target_entropy = 0

target_cols = []

# find the entropy of the target variable (include each class)
for clas in target_classes:                 # find proportions
    prop = data.iloc[:,-1].value_counts()[clas] / num_instances
    class_prop.append(prop)

for i in range(0, num_target_classes):      #calculate entropy of target var
    target_entropy = target_entropy - class_prop[i] * np.log2(class_prop[i])

print('Target var entropy: ' + str(target_entropy))


prev_num_attribute_classes = 0
attr_class_entropy = 0
Attribute_Entropies = pd.DataFrame(index=np.arange(0), columns=np.arange(4 + num_target_classes))

# find entropy of target variable w.r.t. each attribute
for attribute in range(0, num_attributes):                 # for each attribute ...
    attribute_classes = data.iloc[:, attribute].unique()   # find unique attribute classes
    num_attribute_classes = np.count_nonzero(attribute_classes)
    print('\nattribute: ' + str(attribute))
    print('num attribute classes: ' + str(num_attribute_classes))
    print(attribute_classes)

    for clas, i in zip(attribute_classes, range(0,num_attribute_classes)):                         # for each class of an attribute ...
        num_attribute_class_for_target = data.iloc[:,attribute].value_counts()[clas]               # count instances of that class
        Attribute_Entropies.loc[len(Attribute_Entropies), [0, 1, 2]] = [attribute, clas, num_attribute_class_for_target]
        print("HERE")
        print("da entropy:  " + str(attr_class_entropy))
        attr_class_entropy = 0
        
        for clas_1, j in zip(target_classes, range(0,num_target_classes) ):                      # for each target variable class
            print("\nattribute class: " + str(clas) + "      target class: "  + str(clas_1))
            proppy = len(data[(data.iloc[:,attribute]==clas) & (data.iloc[:,-1]==clas_1)])
            print('proppy ' + str(proppy))
            Attribute_Entropies.loc[prev_num_attribute_classes + i, 3 + j] = proppy
            if(proppy !=0):
                attr_class_entropy = attr_class_entropy - proppy/num_attribute_class_for_target * np.log2(proppy/num_attribute_class_for_target)
            else: attr_class_entropy = attr_class_entropy +0
            Attribute_Entropies.iloc[prev_num_attribute_classes + i, -1] = attr_class_entropy
           
    prev_num_attribute_classes = num_attribute_classes

print(Attribute_Entropies)


prev_num_attribute_classes = 0
IG = target_entropy
Attribute_IG = pd.DataFrame(index=np.arange(0), columns=np.arange(2))


# find information gain of each attribute
for attribute in range(0, num_attributes):                 # for each attribute ...
    attribute_classes = data.iloc[:, attribute].unique()   # find unique attribute classes
    num_attribute_classes = np.count_nonzero(attribute_classes)
    Attribute_IG.loc[len(Attribute_IG), 0] = attribute

    print('num attributes: ' + str(num_attributes) + '    num classes:' + str(num_attribute_classes))

    for clas, i in zip(attribute_classes, range(0,num_attribute_classes)):                 # for each attribute ...
        IG = IG - Attribute_Entropies.iloc[prev_num_attribute_classes + i , 2] / num_instances * np.log2(Attribute_Entropies.iloc[prev_num_attribute_classes + i , 2] / num_instances)
        Attribute_IG.iloc[attribute, -1] = IG

    prev_num_attribute_classes = num_attribute_classes

print(Attribute_IG)