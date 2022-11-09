import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
import joblib
def warn(*args,**kwargs):
    pass
warnings.warn =warn
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

Credit_Fraud = pd.read_csv("creditcard_train.csv")
Credit_Fraud_test = pd.read_csv("creditcard_test.csv")
X1 = Credit_Fraud.iloc[:,:-1].values
y1 = Credit_Fraud.iloc[:, 30].values
X2 = Credit_Fraud_test.iloc[:,:-1].values
y2 = Credit_Fraud_test.iloc[:, 30].values
X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size=0.25,random_state=0)
plt.figure()
plt.title('Fraud vs not Fraud')
plt.scatter(X1[:,4],X1[:,29],c=y1,cmap=cmap,edgecolors='k',s=20)
plt.show()
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(pred)
acc = np.sum(pred == y_test)/len(y_test)
print(acc)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
KNN_y = clf.predict(X_test)
result = pd.DataFrame(pred)
result.columns = ["prediction"]
print(result.to_csv("C:\\Information_Science\\למידת מכונה\\KNN_y_ex2.csv"))

