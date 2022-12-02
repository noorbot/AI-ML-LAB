import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
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
trainDataPath = trainData4
testDataPath = testData4

trainDF = pd.read_csv(trainDataPath)
testDF = pd.read_csv(testDataPath)

frames = [trainDF, testDF]
result = pd.concat(frames)


result.replace("xkllklkm", "mjlkm", inplace=True)

listOfAtt = result.keys()
targetVar = listOfAtt[-1]
features = listOfAtt[:-1]

X = result[features].values
y = result[targetVar].values

numOfNeurons = len(features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)

# print("DECISION TREES:")
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train,y_train)
# scores = cross_val_score(clf, X, y)
# print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

print("KNN")
k = 5
clf = neighbors.KNeighborsClassifier(k, weights='uniform')
result.to_csv(r"D:\Users\radam\Desktop\WHATISWRONG.csv", header=True, index = False)
clf.fit(X_train,y_train)
# scores = cross_val_score(clf, X, y)
# print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

# print("Naive Bayes")
# clf = GaussianNB()
# clf.fit(X_train,y_train)
# scores = cross_val_score(clf, X, y)
# print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

# print("ANN")
# #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(numOfNeurons,numOfNeurons,numOfNeurons), random_state=1)
# clf = MLPClassifier(solver='adam', hidden_layer_sizes=(numOfNeurons,numOfNeurons,numOfNeurons), activation='relu', max_iter=1000)
# clf.fit(X_train,y_train)
# scores = cross_val_score(clf, X, y)
# print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

