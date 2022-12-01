from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


data_file_name = "data/lab5/mushroom.csv"     # <-- change file name to match data set

data = pd.read_csv(data_file_name, header = None)          # read from data file and save to pandas DataFrames

target_column = data.iloc[:,-1]
features = data.iloc[:, :-1]
# data[features] = data[features]/data[features].max()

enc = OneHotEncoder(handle_unknown='ignore')
features_t = enc.fit_transform(features)

mlp = MLPClassifier(hidden_layer_sizes=(16,16), activation='relu', solver='adam', max_iter=1000)
mlp.fit(features_t, target_column)


# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(mlp, features_t, target_column, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))