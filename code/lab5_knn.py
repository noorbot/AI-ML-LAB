from sklearn import neighbors
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


data_file_name = "data/lab5/letter_recognition.csv"     # <-- change file name to match data set

data = pd.read_csv(data_file_name, header = None)          # read from data file and save to pandas DataFrames

target_column = data.iloc[:,-1]
features = data.iloc[:, :-1]
# data[features] = data[features]/data[features].max()

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(features, target_column)


# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(knn, features, target_column, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))