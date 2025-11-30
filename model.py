import pandas as pd 
import numpy as np 
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

X,y = fetch_openml(name='mnist_784',version=1,return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
Clf = RandomForestClassifier(n_jobs=-1,random_state=42)
print('training the mode')
print('-'*20)
Clf.fit(X_train,y_train)
print(Clf.score(X_test,y_test))

with open('mnist_model.pkl','bw') as f:
    pickle.dump(Clf,f)
print('the model is saved successfully')    

print('-'*20)