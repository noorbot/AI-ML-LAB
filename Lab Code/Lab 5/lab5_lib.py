import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn import neighbors, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

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

#Set dataset to peform scikit models on
print("A: WDBC")
print("B: LETTER RECOGNITION")
print("C: ECOLI")
print("D: MUSHROOM")
print("E: ROBOT FAILURES - LP5")
userInput = input("Please Select a Dataset: ")

#Ask for user input for specifc dataset
if userInput == "A" or userInput == "a":
    trainDataPath = trainData1
    testDataPath = testData1
    dataset = "A - WDBC"
    ecoliPicked = False
elif userInput == "B" or userInput == "b":
    trainDataPath = trainData2
    testDataPath = testData2
    dataset = "B - LETTER RECOGNITION"
    ecoliPicked = False
elif userInput == "C" or userInput ==  "c":
    trainDataPath = trainData3
    testDataPath = testData3
    dataset = "C - ECOLI"
    ecoliPicked = False
elif userInput == "D" or userInput ==  "d":
    trainDataPath = trainData4
    testDataPath = testData4
    dataset = "D - MUSHROOM"
    ecoliPicked = True
elif userInput == "E" or  userInput == "e":
    trainDataPath = trainData5
    testDataPath = testData5
    dataset = "E - ROBOT FAILURES LP5"
    ecoliPicked = False

#Create training and testing dataframes from .csv files
trainDF = pd.read_csv(trainDataPath)
testDF = pd.read_csv(testDataPath)

#recreate total dataframe
frames = [trainDF, testDF]
result = pd.concat(frames)

#Gather features and target feature
listOfAtt = result.keys()
targetVar = listOfAtt[-1]
features = listOfAtt[:-1]

#For ecoli dataset, must encode categorical data before using ML models from scikit learn
if ecoliPicked == True:
    labelencoder = LabelEncoder()
    for col in features:
        result[col] = labelencoder.fit_transform(result[col])

#Create lists. X - List of feature values, Y - List of target feature values
X = result[features].values
y = result[targetVar].values

#For ANN, use total number of features for dataset the amount of neurons in each layer
numOfNeurons = len(features)

#Create testing and training splits. Don't need test arrays here but due using train_test_split(), we need the compliments of the test size (0.80 in this case)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)

#Print out model accuracies from 5-fold Cross Validation
print("\n DATASET: " + dataset)
print("-----------------------------------------------")

#Each model uses scikit learn to instantiate the machine leanring model. The function fit() is called to then fit the training data to the ML model
#cross_val_score is used to peform 5-fold CV and print the mean and std of the 5 CV scores.
print("DECISION TREES:")
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, y)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

print("KNN:")
k = 5
clf = neighbors.KNeighborsClassifier(k, weights='uniform')
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, y)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

print("Naive Bayes:")
clf = GaussianNB()
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, y)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

print("ANN:")
clf = MLPClassifier(solver='adam', hidden_layer_sizes=(numOfNeurons,numOfNeurons,numOfNeurons), activation='relu', max_iter=1000)
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, y)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))