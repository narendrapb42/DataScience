import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Reading Penguin-dataset
df = pd.read_csv('penguins_size.csv')

#Checking for null values
df.isnull().sum()

df.info()

#Dropping null values
df = df.dropna()
df[df['species']=='Gentoo'].groupby('sex').describe()
df.at[336,'sex'] = 'FEMALE'

#Pairplotting using sns
sns.pairplot(data=df,hue='species')

#Converting features to labels for ML
X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)
y = df['species']

#Train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Importing model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
base_pred = model.predict(X_test)

#Metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,base_pred))

#Visualizing the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(15,17),dpi=200)
plot_tree(model,max_depth=5,filled=True,feature_names=X.columns);