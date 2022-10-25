import pandas as pd
import numpy as np

trainData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\breast-cancer-wisconsin-wLabels-train.csv"
trainData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\letter-recognition-wLabels-train.csv"
trainData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\ecoli-wLabels-train.csv"
trainData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\agaricus-lepiota-wLabels-train.csv"
trainData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\lp5-formatted-wLabels-train.csv"

testData1 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\breast-cancer-wisconsin-wLabels-test.csv"
testData2 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\letter-recognition-wLabels-test.csv"
testData3 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\ecoli-wLabels-test.csv"
testData4 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\agaricus-lepiota-wLabels-test.csv"
testData5 = r"D:\Users\radam\Desktop\ENGR 3150U Lab Files\AI-ML-LAB\Datasets\lp5-formatted-wLabels-test.csv"

lecData = r"D:\Users\radam\Desktop\lecData.csv"


userDataPath = trainData3
trainDataPath = testData3

df = pd.read_csv(userDataPath)
testDF = pd.read_csv(trainDataPath)
columnsNamesArr = df.columns.values
targetAttribute = columnsNamesArr[-1]
print(targetAttribute)

# min_value = df['mcg'].min()
# max_value = df['mcg'].max()
# print(min_value)
# print(max_value)

#df['mcg'] = pd.cut(df['mcg'], bins=np.linspace(min_value, max_value, 21))

# df = df.drop(['Sequence Name'], axis = 1)
df['mcg'] = pd.cut(df['mcg'], bins = 20)
df['gvh'] = pd.cut(df['gvh'], bins = 20)
df['aac'] = pd.cut(df['aac'], bins = 20)
df['alm1'] = pd.cut(df['alm1'], bins = 20)
df['alm2'] = pd.cut(df['alm2'], bins = 20)

# df['mcg'] = pd.cut(df['mcg'], bins = 20)
# df['gvh'] = pd.cut(df['gvh'], bins = 20)
# df['aac'] = pd.cut(df['aac'], bins = 20)
# df['alm1'] = pd.cut(df['alm1'], bins = 20)
# df['alm2'] = pd.cut(df['alm2'], bins = 20)

# df['mcg'] = pd.qcut(df['mcg'], q = 4)
# df['gvh'] = pd.qcut(df['gvh'], q = 4)
# df['aac'] = pd.qcut(df['aac'], q = 4)
# df['alm1'] = pd.qcut(df['alm1'], q = 4)
# df['alm2'] = pd.qcut(df['alm2'], q = 4)
yo = df['mcg'].value_counts()
# df['mcg'] = df.groupby('mcg')['a'].transform('mean')
# df['mcg'] = pd.cut(df['mcg'], bins = 20, include_lowest=True).value_counts()

print(df['mcg'])
print(yo)

# list = {'mcg': 0.0, 'lip': 0.0, 'chg': 0.0}
# attributeWin = max(list, key=list.get) 

# print(attributeWin)

#print(df.head())
# print(yo)
#df.to_csv(r"D:\Users\radam\Desktop\test.csv", header=True, index = False)
#print(df)