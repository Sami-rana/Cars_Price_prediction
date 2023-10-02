import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cars.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
df.drop(columns=['model', 'segment'], inplace=True)
print(df.nunique())

print(df['make'].unique())


def car_make(make):
    if make in ['mazda', 'mg', 'rover', 'alfa-romeo', 'audi', 'peugeot', 'chrysle']:
        return 'Luxury European'
    elif make in ['renault', 'dacia', 'citroen', 'volvo', 'fiat', 'opel', 'seat']:
        return 'Mainstream European'
    elif make in ['gaz', 'aro', 'lada-vaz', 'izh', 'raf', 'bogdan', 'moskvich']:
        return 'Russian/Eastern European'
    elif make in ['toyota', 'nissan', 'asia', 'mitsubishi', 'chery', 'hyundai']:
        return 'Asian'
    elif make in ['oldsmobile', 'gmc', 'chrysler', 'plymouth', 'ford', 'cadillac']:
        return 'American'
    elif make in ['porsche', 'bentley', 'maserati', 'tesla', 'mclaren']:
        return 'Specialty'
    else:
        return 'Other'


df['make_segment'] = df['make'].apply(car_make)

print(df.describe())
print(df.head())

sns.barplot(x=df['make_segment'].unique(), y=df['make_segment'].value_counts())
plt.xticks(rotation=90)
# Show the plot without generating the UserWarning
# plt.show()

fig, ax = plt.subplots(2, 3,figsize=(20, 20))
sns.countplot(x='condition', data=df, ax=ax[0,0])
sns.countplot(x='fuel_type', data=df, ax=ax[0,1])
sns.countplot(x='transmission', data=df, ax=ax[0,2])
sns.countplot(x='color', data=df, ax=ax[1,0])
ax[1,0].tick_params(axis='x',rotation=90)
sns.countplot(x='drive_unit', data=df, ax=ax[1,1])
ax[1,1].tick_params(axis='x',rotation=90)
sns.countplot(x='make_segment', data=df, ax=ax[1,2])
ax[1,2].tick_params(axis='x', rotation=90)

fiq, ax=plt.subplots(2,2,figsize=(20,10))
sns.histplot (df['year'], ax=ax[0,0], bins = 50)
sns.histplot(df['priceUSD'], ax=ax[0,1])
sns.histplot(df['mileage(kilometers)'], ax=ax[1,0], bins = 100)
sns.histplot(df['volume(cm3)'], ax=ax[1,1], bins = 100)

df= df[df['year']>1980]
demodf = df.groupby('make')['priceUSD'].mean().reset_index()
demodf = demodf.sort_values(by='priceUSD', ascending=False).head(10)
#b Bar Plot
plt.figure(figsize=(8,5))
sns.barplot(y='make', x='priceUSD', data=demodf)
plt.xticks(rotation=90)
plt.title('Top 10 Most Expensive Car Brands')
plt.ylabel('Car Brand')
plt.xlabel('Price in USD')
# plt.show()

sns.lineplot(x = 'year', y = 'priceUSD', data = df, hue = 'condition' )
plt.title("Price Of Cars By Year and Condition")
# plt.show()

sns.lineplot(x = 'year', y = 'priceUSD', data = df, hue = 'fuel_type' )
plt.title("Price Of Cars By Year and Fuel Type")
# plt.show()

sns.lineplot(x = 'year', y = 'priceUSD', data = df, hue = 'drive_unit')
plt.title("Price Of Cars By Year and DRIVE UNIT")
# plt.show()

sns.lineplot(x = 'year', y = 'priceUSD', data = df, hue = 'make_segment' )
plt.title("Price Of Cars By Year and Brand Segment")
# plt.show()

df.isnull().sum()
df.dropna(inplace = True)
df.drop(columns = ('make'),inplace = True)

from sklearn.preprocessing import LabelEncoder
cols = ['condition', 'fuel_type', 'transmission','color', 'drive_unit','make_segment']
le = LabelEncoder()

for col in cols:
    le.fit(df[col])
    df[col] = le.transform(df[col])
    print(col, df[col].unique())


plt.figure(figsize = (12, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

from scipy import stats
z = np.abs(stats.zscore(df))
threshold = 3

cols = ['year','mileage(kilometers)', 'volumn(cm3)']

df = df[(z < 3).all(axis=1)]

from sklearn.model_selection import train_test_split
X = df.drop(columns=['priceUSD'])
y = df['priceUSD']
# X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['priceUSD']))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

from sklearn.model_selection import GridSearchCV
#parameters for grid search
params = {
'max_depth': [2,4,6,8],
'min_samples_split': [2,4,6,8],
'min_samples_leaf': [1,2,3,4],
'max_features': ['auto', 'sqrt', 'log2'],
'random_state': [0,42]
}
# Grid Search Object
grid = GridSearchCV(dtr, param_grid=params, cv=5, verbose=1, n_jobs=-1)
#fitting the grid search
grid.fit(X_train, y_train)
#best parameters
print(grid.best_params_)

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_depth=8, max_features=None, min_samples_leaf=4, random_state=0)

# Fitting the model
dtr.fit(X_train, y_train)
dtr.score(X_train, y_train)
y_pred = dtr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('R2 Score: ', r2_score(y_test, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)))

feat_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': dtr.feature_importances_})
feat_df = feat_df.sort_values(by='Importance', ascending=False)
print(feat_df)

sns.set_style('darkgrid')
plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y = 'Feature', data = feat_df)
plt.title("Feature Importance")
# plt.show()
