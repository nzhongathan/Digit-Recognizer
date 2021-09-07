#importing libraries
import numpy as np 
import pandas as pd 
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
fig_dims = (20,10)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import os

#preparing the data
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')

train.head(5)

x = train.copy()
x=x.drop(columns=['label'])
y = train['label']

#normalizing the data
x = x / 255.0

x.iloc[0].max()

#K-nearest neighbors classifier
train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=32)

model = KNeighborsClassifier (n_neighbors=5,weights = 'distance')
model.fit(train_x,train_y)
preds = model.predict(val_x)
score = accuracy_score(preds,val_y)*100
print('Error: ',score) #Error:  96.5904761904762

#XGBoost classifier
xgboost = XGBClassifier()
xgboost.fit(train_x,train_y)
preds = xgboost.predict(val_x)
score = accuracy_score(preds,val_y)*100
print('Error: ',score) #Error:  97.23809523809524

#Naive Bayes Classifier
  #Multinomial
nb = MultinomialNB()
nb.fit(train_x,train_y)
preds = nb.predict(val_x)
score = accuracy_score(preds,val_y)* 100
print('Error: ', score) #Error:  82.77142857142857
  #Gaussian
gnb = GaussianNB()
gnb.fit(train_x,train_y)
preds = gnb.predict(val_x)
score = accuracy_score(preds,val_y)* 100
print('Error: ', score) #Error:  56.57142857142857
  #Bernoulli
bnb = BernoulliNB()
bnb.fit(train_x,train_y)
preds = bnb.predict(val_x)
score = accuracy_score(preds,val_y)* 100
print('Error: ', score) #Error:  83.4095238095238

#Creating the Submission
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

test = test/255.0 #Remember the normalize the test data like we did for the training data

predictions = xgboost.predict(test)
submission['Label'] = predictions
submission.to_csv('submission.csv',index=False)
