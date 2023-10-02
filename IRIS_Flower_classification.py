import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn import metrics

df = pd.read_csv('IRIS.csv')
print(df.head())

print(df.shape)
print(df.info())

print(df.isnull().sum())
print(df.describe())

plt.figure(figsize=(8, 6))
sns.countplot(x='species', data=df)
plt.xlabel('Species', fontsize=14)
plt.show()

df['species'].value_counts()
sns.set(rc={'figure.figsize': (10, 8)})
sns.distplot(df["sepal_length"], kde=True, color="red", bins=10)

plt.figure(figsize=(8, 6))
sns.scatterplot(x="sepal_length", y=df.index, data=df)
plt.show()

sns.set(rc={"figure.figsize": (8, 6)})
sns.distplot(df["sepal_width"], kde=True, color="navy", bins=10)

plt.figure(figsize=(10, 8))
sns.scatterplot(x="sepal_width", y=df.index, data=df)
plt.show()

sns.set(rc={'figure.figsize': (8, 6)})
sns.distplot(df['petal_length'], kde=True, color="blue", bins=10)

plt.figure(figsize=(8, 8))
sns.scatterplot(x="petal_length", y=df.index, data=df)
plt.show()

sns.set(rc={"figure.figsize": (10, 8)})
sns.distplot(df["petal_width"], kde=True, color="orange", bins=10)

plt.figure(figsize=(10, 8))
sns.scatterplot(x="petal_width", y=df.index, data=df)
plt.show()

duplicate = df.duplicated()
print(duplicate.sum())

df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

print(df.isnull().sum())

num_cols = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(16, 14))
sns.boxplot(num_cols)
plt.show()


def remove_outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


for i in num_cols.columns:
    lower_range, upper_range = remove_outlier(df[i])
    df[i] = np.where(df[i] > upper_range, upper_range, df[i])
    df[i] = np.where(df[i] < lower_range, lower_range, df[i])

num_cols = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(14, 12))
sns.boxplot(num_cols)
plt.show()

print(df.info())

plt.figure(figsize=(18, 16))
sns.histplot(data=df, x="sepal_length", hue='species', kde=True, bins=20)
plt.title('species By Sepal_Length')
plt.xlabel('Sepal_length')
plt.ylabel('Count')
plt.legend(title='species', loc='upper right', labels=['No Species', 'species'])
plt.show()

plt.figure(figsize=(18, 16))
sns.histplot(data=df, x="sepal_width", hue='species', kde=True, bins=20)
plt.title('species By Sepal_width')
plt.xlabel('Sepal_width')
plt.ylabel('Count')
plt.legend(title='species', loc='upper right', labels=['No Species', 'species'])
plt.show()

plt.figure(figsize=(18, 16))
sns.histplot(data=df, x="petal_length", hue='species', kde=True, bins=20)
plt.title('species By petal_Length')
plt.xlabel('petal_length')
plt.ylabel('Count')
plt.legend(title='species', loc='upper right', labels=['No Species', 'species'])
plt.show()

plt.figure(figsize=(18, 16))
sns.histplot(data=df, x="petal_width", hue='species', kde=True, bins=20)
plt.title('species By petal_width')
plt.xlabel('petal_width')
plt.ylabel('Count')
plt.legend(title='species', loc='upper right', labels=['No Species', 'species'])
plt.show()

sns.scatterplot(x="sepal_length", y="petal_length", data=df, hue="species")
plt.show()

sns.scatterplot(x="sepal_width", y="petal_width", data=df, hue="species")
plt.show()

sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)

sns.pairplot(df, hue="species")
plt.show()


def plots(num_cols, variable):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)  # num_cols[variable].hist()
    sns.distplot(num_cols[variable], kde=True, bins=10)
    plt.title(variable)
    plt.subplot(1, 2, 2)
    stats.probplot(num_cols[variable], dist="norm", plot=pylab)
    plt.title(variable)
    plt.show()


for i in num_cols.columns:
    plots(num_cols, i)

X = df.iloc[:,:4]
Y = df['species']
X.head()

mi_score = mutual_info_classif(X,Y)
mi_score = pd.Series(mi_score)
mi_score.index = X.columns
mi_score.sort_values(ascending=True)

mi_score.sort_values(ascending=True).plot.bar(figsize=(12,10))

train_data,test_data,train_label,test_label = train_test_split(X,Y,test_size=0.2,random_state=0)
print("train_data :",train_data.shape)
print("train_label :",train_label.shape)
print("test_data :",test_data.shape)
print("test_label :",test_label.shape)

sc = StandardScaler()
train_data_sc = sc.fit_transform(train_data)
test_data_sc = sc.fit_transform(test_data)

print(train_data_sc)

model_lr = LogisticRegression().fit(train_data_sc, train_label)

y_pred_1 = model_lr.predict(test_data_sc)
print(y_pred_1)

accuracy_score(y_pred_1,test_label)

confusion_matrix(y_pred_1,test_label)

confusion_matrix(y_pred_1,test_label)

confusion_matrix(y_pred_1,test_label)

print(classification_report(y_pred_1,test_label))

model_rf = RandomForestClassifier().fit(train_data_sc,train_label)

y_pred_2 = model_rf.predict(test_data_sc)

print("Train Data Accuracy :",(model_rf.score(train_data_sc,train_label)))
print("Test Data Accuracy :",(accuracy_score(y_pred_2,test_label)))

confusion_matrix(y_pred_2,test_label)

print(classification_report(y_pred_2,test_label))

# KNN Model
model_knn = KNeighborsClassifier(n_neighbors=3).fit(train_data_sc,train_label)

y_pred_3 = model_knn.predict(test_data_sc)

print("Test Data Accuracy :",(accuracy_score(y_pred_3,test_label)))

confusion_matrix(y_pred_3,test_label)

print(classification_report(y_pred_3,test_label))











