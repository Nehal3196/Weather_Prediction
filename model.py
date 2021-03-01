#Importing the libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

data = pd.read_csv('Weather1.csv')

data['Temperature'].fillna(data['Temperature'].mean(), inplace=True)

x = data.drop(['Event','Date'],axis =1).values
y = data['Event'].values


#Since we have a very small dataset, we will train our model with all availabe data.
model_dt = DecisionTreeClassifier(random_state=1, max_depth=4, criterion = "gini")

#Fitting model with trainig data
model_dt.fit(x,y)

# Saving model to disk
pickle.dump(model_dt, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[25,5]]))