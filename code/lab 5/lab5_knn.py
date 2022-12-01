from sklearn import neighbors
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


data_file_name = "data/lab5/lp5.csv"                              # <-- change file name to match data set

data = pd.read_csv(data_file_name, header = None)                 # read from data file and save to pandas DataFrames

target_column = data.iloc[:,-1]                                   # separate target column and feature columns
features = data.iloc[:, :-1]

knn = neighbors.KNeighborsClassifier(n_neighbors=5)               # create knn  model
knn.fit(features, target_column)                                  # fit model to data

cv_scores = cross_val_score(knn, features, target_column, cv=5)   # compute 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))   # take mean of score and print it
