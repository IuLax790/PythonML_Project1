import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
Salary = pd.read_csv("salary_train.csv")
Salary_test = pd.read_csv("salary_test.csv")
print(Salary_test)
X = Salary.iloc[:, :-1].values

y = Salary.iloc[:, 20].values

imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
X[:,[3]]=imp_median.fit_transform(X[:,[3]])
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:,[9]]=imp_mean.fit_transform(X[:,[9]])
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:,[14]]=imp_mean.fit_transform(X[:,[14]])
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:,[19]]=imp_mean.fit_transform(X[:,[19]])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(sparse=False)
Z = onehotencoder.fit_transform(X[:,[6]])
X=np.hstack((X[:,:6],Z)).astype('float')
X = X[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)
import statsmodels.api as sm
import statsmodels.tools.tools as tl
X = tl.add_constant(X)

SL = 0.05
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

numVars = len(X_opt[0])
for i in range(0,numVars):
    regressor_OLS = sm.OLS(y,X_opt).fit()
    max_var = max(regressor_OLS.pvalues).astype(float)
    if max_var > SL:
        new_Num_Vars = len(X_opt[0])
        for j in range(0,new_Num_Vars):
            if (regressor_OLS.pvalues[j].astype(float)==max_var):
                X_opt = np.delete(X_opt,j,1)
print(regressor_OLS.summary())

print(y_pred)
print(regressor_OLS.pvalues)
print(regressor_OLS.params)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
plt.figure(figsize=(10,8))
plt.scatter(y_test,y_pred)
plt.xlabel('Variables')
plt.ylabel('Salary')
plt.title('Salary Prediction')

print(plt.show())
predicted_y = regressor.predict
print(predicted_y)
y_pred = regressor.predict(X_test)
result = pd.DataFrame(y_pred)
result.columns = ["prediction"]
