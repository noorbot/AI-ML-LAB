import pandas as pd
import numpy as np

file_name = "results/letter_recognition_train_data.csv"         # <-- change file name to match data set

# read from data file and save to pandas DataFrame 'data'
#data = pd.read_csv(file_name, header = None)
data = pd.DataFrame([[1, 2, 'A'], 
[4, 5, 'B'], 
[5, 8, 'C'],
[2, 7, 'A'], 
[5, 8, 'A'],
[2, 5, 'B'], 
[2, 8, 'C'],
[4, 5, 'A'], 
[5, 9, 'C'],
])

print(data.head())

# count the number of attributes
num_attributes = data.shape[1] - 1
num_instances = data.shape[0]

target_classes = data.iloc[:, -1].unique()
print(target_classes)
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

print(target_entropy)



Attribute_Entropies = pd.DataFrame(index=np.arange(1), columns=np.arange(4 + num_target_classes))

# find entropy of target variable w.r.t. each attribute
for attribute in range(0, num_attributes):                 # for each attribute ...
    attribute_classes = data.iloc[:, attribute].unique()   # find unique attribute classes
    num_attribute_classes = np.count_nonzero(attribute_classes)
    print('attribute: ' + str(attribute))
    print('num attribute classes: ' + str(num_attribute_classes))
    print(attribute_classes)

    for clas, i in zip(attribute_classes, range(0,num_attribute_classes)):                         # for each class of an attribute ...
        num_attribute_class_for_target = data.iloc[:,attribute].value_counts()[clas]               # count instances of that class
        Attribute_Entropies.loc[attribute + i, 0] = attribute
        Attribute_Entropies.loc[attribute + i, 1] = clas
        Attribute_Entropies.loc[attribute + i, 2] = num_attribute_class_for_target
        
        for clas_1, j in zip(target_classes, range(0,num_target_classes) ):                      # for each target variable class
            #proppy = data.iloc[:,attribute].value_counts()[clas_1]    # number of time this attribute with this target variable class
            #Attribute_Entropies.loc[attribute + i, 3 + j] = proppy
            print("yeah this is where i need to fix. proppy")

print(Attribute_Entropies)