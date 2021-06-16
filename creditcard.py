#importing the libraries
import numpy as np
import pandas as pd

#load the dataset
dataset = pd.read_csv('creditcard.csv')

#understanding the data
dataset.info()
dataset.isnull().sum()
legit=dataset[dataset.Class==0]
fraud=dataset[dataset.Class==1]
legit.Amount.describe()
fraud.Amount.describe()

#compare the values for both transaction
a=dataset.groupby('Class').mean()

#dealing with unbalanced data
#Build a sample dataset containing similar distribution of normal transaction and fraud transaction
legit_sample=legit.sample(n=492)
new_dataset = pd.concat([legit_sample,fraud],axis = 0)
new_dataset['Class'].value_counts()

#mean tells us that we got a good dataset
b=new_dataset.groupby('Class').mean()

#splliting the dataset
X=new_dataset.drop(columns='Class',axis=1)
y=new_dataset['Class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

#Training the model
from sklearn.linear_model import LogisticRegression
classifier =  LogisticRegression()
classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

#Comparing the correct and wrong prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,ypred)


