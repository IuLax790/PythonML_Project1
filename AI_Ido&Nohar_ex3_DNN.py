import clf as clf
import joblib
import pandas as pd
import numpy as np
import skf as skf

DataSet = pd.read_csv("C:\\Information_Science\\למידת מכונה\\spam_train.csv",encoding="latin-1")
Dataset_Test = pd.read_csv("C:\\Information_Science\\למידת מכונה\\spam_test.csv",encoding="latin-1")
DataSet['v1']=DataSet["v1"].map({"ham":0,"spam":1})
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x= DataSet["v2"]
y= DataSet["v1"]
print(x)
print(y)
x= cv.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(pred)
res= model.score(x_test,y_test)
res*=100
print(res)
print(joblib.dump(pred,open("C:\\Information_Science\\למידת מכונה\\spam_prediction.pkl","wb")))
print(joblib.dump(cv,open("C:\\Information_Science\\למידת מכונה\\vectorizer.pkl","wb")))
msg = "You won a million dollar. fill your bank account in the link below to get the prize"
data = [msg]
vect = cv.transform(data).toarray()
result = model.predict(vect)
print(result)
msg = "Hi, this is Ted from ML Course.I hope you're all feel well. here are the assignment for the next week"
data = [msg]
vect = cv.transform(data).toarray()
result = model.predict(vect)
print(result)
result = pd.DataFrame(pred)
result.columns = ["prediction"]
print(result.to_csv("C:\\Information_Science\\למידת מכונה\\DNN_y.csv"))