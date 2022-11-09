import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")
def warn(*args,**kwargs):
    pass
Credit_Fraud = pd.read_csv("C:\\Information_Science\\למידת מכונה\\creditcard_train.csv")
Credit_Fraud_test = pd.read_csv("C:\\Information_Science\\למידת מכונה\\creditcard_test.csv")

X = Credit_Fraud.iloc[:,[2,29]].values
y = Credit_Fraud.iloc[:, 30].values
colormap = plt.cm.viridis
plt.figure(figsize=(30,30))
plt.title('Pearson correlation of attributes',y=1.05,size=19)
sns.heatmap(Credit_Fraud.corr(),linewidths=0.1,vmax=1.0,
            square=True,cmap=colormap,linecolor='white',annot=True)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svc = SVC()
svc.fit(X_train,y_train)
pred = svc.predict(X_test)
print(svc.score(X_train,y_train))
print(svc.score(X_test,y_test))

from sklearn import metrics
print(metrics.confusion_matrix(y_test,pred))

from imblearn.over_sampling import SMOTE
over_sample = SMOTE()
x_smote,y_smote = over_sample.fit_resample(X_train,y_train)
print(sns.countplot(y_smote))
print(sns.countplot(pred))
result = pd.DataFrame(pred)
result.columns = ["prediction"]
print(result.to_csv("C:\\Information_Science\\למידת מכונה\\SVM_y_ex2.csv"))