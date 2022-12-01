import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import neighbors, tree
from sklearn.naive_bayes import GaussianNB

#File paths for training datasets
trainData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\wdbc_train_data.csv"
trainData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\letter-recognition-wLabels-train.csv"
trainData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\ecoli_train_data.csv"
trainData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\agaricus-lepiota-wLabels-train.csv"
trainData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\lp5_train_data.csv"

#File paths for testing datasets
testData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\wdbc_test_data.csv"
testData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\letter-recognition-wLabels-test.csv"
testData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\ecoli_test_data.csv"
testData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\agaricus-lepiota-wLabels-test.csv"
testData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\Lab 2 Datasets\lp5_test_data.csv"

#Debugging datasets
lecData = r"D:\Users\radam\Desktop\lecData.csv"
lecDataTest = r"D:\Users\radam\Desktop\lecDataTest.csv"
testData = r"D:\Users\radam\Desktop\test.csv"
test = r"D:\Users\radam\Desktop\wdbc_test_data.csv"
train = r"D:\Users\radam\Desktop\wdbc_train_data.csv"

#Set dataset to build decision tree from and test
trainDataPath = trainData2
testDataPath = testData2

#For debugging and testing the program
# userDataPath = lecData
# #testDataPath = testData
# testDataPath= lecDataTest

#Debu9g dataframes using csv files
# trainDF = pd.read_csv(userDataPath)
# testDF = pd.read_csv(testDataPath)

#Create dataframes using csv files
# df = pd.read_csv(ogData1)

# from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
# mlp.fit(X_train,y_train)

# predict_train = mlp.predict(X_train)
# predict_test = mlp.predict(X_test)

trainDF = pd.read_csv(trainDataPath)
testDF = pd.read_csv(testDataPath)

frames = [trainDF, testDF]
result = pd.concat(frames)

listOfAtt = result.keys()
targetVar = listOfAtt[-1]
features = listOfAtt[:-1]

X = result[features].values
y = result[targetVar].values

numOfNeurons = len(features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)

# print("1: Decision Trees")
# print("2: KNN")
# print("3: Naive Bayes")
# print("4: ANN")
# userInput = input("Please Select a Model: ")
# wrongFlag = False

# if userInput == 1:
#     clf = tree.DecisionTreeClassifier()
#     clf.fit(X_train,y_train)
# elif userInput == 2:
#     k = 5
#     clf = neighbors.KNeighborsClassifier(k, weights='uniform')
#     clf.fit(X_train,y_train)
# elif userInput == 3:
#     clf = GaussianNB()
#     clf.fit(X_train,y_train)
# elif userInput == 4:
#     clf = MLPClassifier(hidden_layer_sizes=(8,8,8,8), activation='relu', solver='adam', max_iter=500)
#     clf.fit(X_train,y_train)
# else:
#     wrongFlag = True

print("DECISION TREES:")
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, y)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

print("KNN")
k = 5
clf = neighbors.KNeighborsClassifier(k, weights='uniform')
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, y)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

print("Naive Bayes")
clf = GaussianNB()
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, y)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

print("ANN")
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(numOfNeurons,numOfNeurons,numOfNeurons), random_state=1)
clf = MLPClassifier(solver='adam', hidden_layer_sizes=(numOfNeurons,numOfNeurons,numOfNeurons), activation='relu', max_iter=1000)
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, y)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))





# predict_train = mlp.predict(X_train)
# predict_test = mlp.predict(X_test)
# print(confusion_matrix(y_train,predict_train))
# print(classification_report(y_train,predict_train))

# #print(scores)
# clf.score(X_test, y_test)
# print(features)
# print (targetVar)
# result.to_csv(r"D:\Users\radam\Desktop\OPEN-wLabels-train.csv", header=True, index = False)
# print(df.shape)

# scores = cross_val_score(clf, X, y)
# print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))