from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


train_file_name = "results/mushroom_train_data.csv"     # <-- change file names to match data set (use binned data for numerical datasets)
test_file_name = "results/mushroom_test_data.csv"

train_data = pd.read_csv(train_file_name, header = None)          # read from data file and save to pandas DataFrames
test_data = pd.read_csv(test_file_name, header=None)

target_column = train_data.iloc[:,-1]
features = 

X = 
y = 
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(clf, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))