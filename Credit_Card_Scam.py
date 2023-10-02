#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics._plot.roc_curve import auc

# In[2]:


df = pd.read_csv('creditcard.csv')
df.head()

# In[3]:


df.shape

# In[4]:


df.info()

# In[5]:


df.isnull().sum()

# In[6]:


df.describe()

# In[7]:


df.info()

# In[8]:


sns.set(rc={'figure.figsize': (8, 6)})
sns.distplot(df['V1'], kde=True, color='red', bins=10)

# In[9]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='V1', y=df.index, data=df)
plt.show

# In[10]:


sns.set(rc={"figure.figsize": (8, 6)})
sns.distplot(df["V2"], kde=True, color='green', bins=10)

# In[11]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='V2', y=df.index, data=df)
plt.show()

# In[12]:


sns.set(rc={'figure.figsize': (8, 6)})
sns.distplot(df['V3'], kde=True, color='magenta', bins=10)

# In[13]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x="V3", y=df.index, data=df)
plt.show()

# In[14]:


sns.set(rc={"figure.figsize": (8, 6)})
sns.distplot(df["V4"], kde=True, color='black', bins=10)

# In[15]:


plt.figure(figsize=(8, 8))
sns.scatterplot(x="V4", y=df.index, data=df)
plt.show()

# In[16]:


sns.set(rc={"figure.figsize": (8, 6)})
sns.distplot(df["V5"], kde=True, color='orange', bins=10)

# In[17]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x="V5", y=df.index, data=df)
plt.show()

# In[18]:


sns.set(rc={"figure.figsize": (8, 6)})
sns.distplot(df["Amount"], kde=True, color='cyan', bins=10)

# In[19]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x="Amount", y=df.index, data=df)
plt.show()

# In[20]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette='mako')

# In[21]:


df['Class'].value_counts()

# In[22]:


duplicate = df.duplicated()
print(duplicate.sum())

# In[23]:


df.drop_duplicates(inplace=True)

# In[24]:


duplicate = df.duplicated()
print(duplicate.sum())

# In[25]:


df.isnull().sum()

# In[26]:


plt.figure(figsize=(24, 10))
sns.boxplot(df)
plt.show()


# In[27]:


def remove_outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 + IQR)
    return lower_range, upper_range


# In[28]:


df.columns.get_loc("Class")

# In[29]:


for i in df.columns:
    if df.columns.get_loc(i) == 30:
        pass
    else:
        lower_range, upper_range = remove_outlier(df[i])
        df[i] = np.where(df[i] > upper_range, upper_range, df[i])
        df[i] = np.where(df[i] < lower_range, lower_range, df[i])

# In[30]:


plt.figure(figsize=(24, 10))
sns.boxplot(df)
plt.show()

# In[31]:


sns.boxplot(df['V1'])
plt.show()

# In[32]:


sns.boxplot(df['V2'])
plt.show()

# In[33]:


sns.boxplot(df['Amount'])
plt.show()

# In[34]:


sns.boxplot(df['Class'])
plt.show()

# In[35]:


df['Class'].value_counts()

# In[36]:


for i in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=i, hue='Class', kde=True, bins=20)
    plt.xlabel('i')
    plt.ylabel('Count')
    plt.legend(title='Class', loc='upper right', labels=['No Class', 'Class'])
    plt.show()

# In[37]:


plt.figure(figsize=(24, 12))
sns.heatmap(df.corr(), cmap='YlGnBu', annot=True)
plt.show()


# In[38]:


# sns.pairplot(df,hue ='Class')
# plt.show()


# In[39]:


def plots(df, variable):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    df[variable].hist()
    sns.distplot(df[variable], kde=True, bins=10)
    plt.title(variable)
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist='norm', plot=pylab)
    plt.title(variable)
    plt.show()


for i in df.columns:
    plots(df, i)

# In[40]:


X = df.iloc[:, 1:30]
Y = df['Class']

# In[41]:


X.head()

# In[42]:


mi_score = mutual_info_regression(X, Y)
mi_score = pd.Series(mi_score)
mi_score.index = X.columns
mi_score.sort_values(ascending=True)

# In[43]:


mi_score.sort_values(ascending=True).plot.bar(figsize=(20, 10))

# In[44]:


train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.3, random_state=0)

# In[45]:


print("train_data :", train_data.shape)
print("train_label:", train_label.shape)
print("test_data:", test_data.shape)
print("test_label:", test_label.shape)

# In[46]:


sc = StandardScaler()
train_data_sc = sc.fit_transform(train_data)
test_data_sc = sc.fit_transform(test_data)

# In[47]:


train_data_sc

# In[48]:


pc = PCA()
train_data_sc_pc = pc.fit_transform(train_data_sc)
test_data_sc_pc = pc.fit_transform(test_data_sc)

# In[49]:


explained_variance = pc.explained_variance_ratio_
print("Explained Variance Ratios:", explained_variance)

# In[50]:


# CALCLATE CUMULATIVE SUM OF EXPLAINED VARIANCE RATIO
cumulative_variance = np.cumsum(explained_variance)

# PLOT THE SCREE PLOT OR CUMULATIVE EXPLAINED VARIANCE PLOT
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='-')
plt.xlabel('Number of Comonents')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot or Cumulative Explained Variance Plot')
plt.grid(True)
plt.show()

# In[52]:


train_data_sc_pc.shape

# In[54]:


cumulative_variance = np.cumsum(explained_variance)

desired_variance = 0.95
num_components = np.argmax(cumulative_variance >= desired_variance) + 1
print(f"\nNumber of Components selected:{num_components}")

# In[56]:


train_data_sc_pc_select = train_data_sc_pc[:, :num_components]
test_data_sc_pc_select = test_data_sc_pc[:, :num_components]

# In[57]:


print("train_data :", train_data_sc_pc_select.shape)
print("test_data :", test_data_sc_pc_select.shape)

# In[59]:


model_lr = LogisticRegression().fit(train_data_sc_pc, train_label)

# In[61]:


model_lr.score(train_data_sc_pc_select, train_label)

# In[ ]:


y_pred_1 = model_lr.predict(test_data_sc_pc_select)
y_pred_1

# In[ ]:


print(accuracy_score(y_pred_1, test_label))

# In[ ]:


confusion_matrix(y_pred_1, test_label)

# In[ ]:


print(classification_report(y_pred_1, test_label))

# In[ ]:


fpr, tpr, thresholds = roc_curve(test_label, y_pred_1)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='==')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('receiver Operating characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
