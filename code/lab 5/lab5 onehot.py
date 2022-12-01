from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


data_file_name = "data/lab5/lp5.csv"                              # <-- change file name to match data set

data = pd.read_csv(data_file_name, header = None)                 # read from data file and save to pandas DataFrames

target_column = data.iloc[:,-1]                                   # separate target column and feature columns
features = data.iloc[:, :-1]

enc = OneHotEncoder(handle_unknown='ignore')                      # create one hot encoder 
features_t = enc.fit_transform(features)                          # apply encoder to features column of the data to encode

# create multi layer perceptron classifer and fit it to data
mlp = MLPClassifier(hidden_layer_sizes=(16,16), activation='relu', solver='adam', max_iter=1000)
mlp.fit(features_t, target_column)

cv_scores = cross_val_score(mlp, features_t, target_column, cv=5) # compute 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))   # take mean of score and print it