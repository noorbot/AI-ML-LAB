from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


data_file_name = "data/lab5/wdbc.csv"                             # <-- change file name to match data set

data = pd.read_csv(data_file_name, header = None)                 # read from data file and save to pandas DataFrames

target_column = data.iloc[:,-1]                                   # separate target column and feature columns
features = data.iloc[:, :-1]

# create multu layer perceptron classifier and fit it to data
mlp = MLPClassifier(hidden_layer_sizes=(16,16), activation='relu', solver='adam', max_iter=1000)
mlp.fit(features, target_column)

cv_scores = cross_val_score(mlp, features, target_column, cv=5)   # compute 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))   # take mean of score and print it