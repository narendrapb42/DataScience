import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')
#To describe dataset info
df.describe()

#Display total count values
df['Species'].value_counts()

#pairplotting the dataset
sns.pairplot(data=df,hue='Species')

#create features and labels
X = df.drop('Species',axis=1)
y = df['Species']

#training and splitting data
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Scaling the values
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#Importing necessary models
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

#Fitting the model
log_model = LogisticRegression(solver='saga',multi_class='ovr',max_iter=5000)
penalty = ['l1','l2','elasticnet']
l1_ratio = np.linspace(0,1,20)
C = np.logspace(0,10,20)
param_grid = {'penalty':penalty,'l1_ratio':l1_ratio,'C':C}

grid_model = GridSearchCV(log_model,param_grid=param_grid)
grid_model.fit(scaled_X_train,y_train)

#Calculating metric score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(grid_model.best_params_)

y_pred = grid_model.predict(scaled_X_test)
print(y_pred)
