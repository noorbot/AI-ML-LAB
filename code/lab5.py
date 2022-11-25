from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


data_file_name = "data/lab5/ecoli.csv"     # <-- change file name to match data set

data = pd.read_csv(data_file_name, header = None)          # read from data file and save to pandas DataFrames

target_column = data.iloc[:,-1]
features = data.iloc[:, :-1]

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(features, target_column)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(mlp, features, target_column, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))