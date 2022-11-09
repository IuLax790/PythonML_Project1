
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

import warnings
def warn(*args,**kwargs):
    pass
warnings.warn = warn
# Importing the dataset
X = pd.read_csv("C:\\Information_Science\\למידת מכונה\\Titanic_train.csv")
y=X.pop("Survived")
print(X)
print(y)
X["Age"].fillna(X.Age.mean(),inplace=True)
numeric_variables = list(X.dtypes[X.dtypes !="object"].index)
X[numeric_variables].head()
model = RandomForestRegressor(n_estimators=100,oob_score=True,random_state=0)
model.fit(X[numeric_variables],y)
def describe_categorical(X):
    """just like .describe(), but returns the results for categorical

    variables only"""
    from IPython.display import display,HTML
    display(HTML(X[X.columns[X.dtypes=="object"]].describe().to_html()))
print(describe_categorical(X))
X.drop(["Name","Ticket","PassengerId"],axis=1,inplace=True)
def clean_cabin(X):
    try:
        return X[0]
    except TypeError:
        return "None"
X["Cabin"]=X.Cabin.apply(clean_cabin)
categorical_variables = ['Sex','Cabin','Embarked']
for v in categorical_variables:
    X[v].fillna("Missing",inplace=True)
    dummies = pd.get_dummies(X[v],prefix=v)
    X = pd.concat([X,dummies],axis=1)
    X.drop([v],axis=1,inplace=True)
def print_all(X,max_rows=10):
    from IPython.display import display,HTML
    display(HTML(X.to_html(max_rows=max_rows)))
print(print_all(X))



regressor=RandomForestRegressor(n_estimators=3,oob_score=True, random_state=0)
regressor.fit(X,y)
y_oob = regressor.oob_prediction_
print("C-Stat:",roc_auc_score(y,y_oob))
model = regressor.feature_importances_
feature_importances = pd.Series(regressor.feature_importances_,index=X.columns)
feature_importances.plot(kind='barh',figsize=(7,6))
feature_importances.sort_index()
plt.title('feature importances')
print(plt.show())

predict = regressor.predict(X)
result = pd.DataFrame(predict)
result.columns = ["prediction"]
print(result.to_csv("C:\\Information_Science\\למידת מכונה\\RF_y_ex1.csv"))

X_grid = np.arange(min(X), max(X), 0.1) #we get a vector
X_grid = X_grid.reshape((len(X_grid), 1)) #we need a matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree')
plt.xlabel('Position level')
plt.ylabel('Salary')
print(plt.show())


my_names=[k for k in Titanic.keys()] #get the names of the columns
my_names=my_names[1:-1] #remove the name of the dependent variable (and position)
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=my_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
z=regressor.predict([[5.8]])
print(z)