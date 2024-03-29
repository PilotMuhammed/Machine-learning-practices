#The Euclidean Distance  between two points in a plane is the length of the straight line that connects those points. 
#Write a python program to calculate the Euclidean distance between two points in 2D space(two features).
import math
#math module  is used for mathematical functions and constants
print("Enter the first point A")
#map(int,input)  function is used to convert string input into integer and split  by space
x1, y1  = map(int, input().split())
print("Enter the second point B")
x2, y2 = map(int, input().split())
dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
print("The Euclidean Distance is " + str(dist))

#HW1: what about 3 or more features?

# knn - method 1
#numpy  module  is imported as np which used  for numerical operations
import numpy as np
#sklearn.neighbors is used  for k nearest neighbours algorithm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
df=pd.read_csv("data.csv")
#df.iloc[]  is used to get data from csv file and the parmeter in [] represnt s row number
#df.iloc[0:6,:-1] represent  rows from 0 to 5 and columns from 0 to -1 (not include last column)
#df.iloc[0:6,-1]  represents last column of rows from  0 to  5
#df.iloc[5:,:-1] represents  all rows after 5th row and all columns except last one
x_train=df.iloc[0:6,:-1]
print("------ Display the x_train ------")
print(x_train)
y_train=df.iloc[0:6,-1]
x_test=df.iloc[5:,:-1]

# knn1=KNeighborsClassifier(n_neighbors=5) represent   that we are using 5 nearest neighbour classifier
knn1=KNeighborsClassifier(n_neighbors=5)
# Fit the model to the training data which means  train the model with our dataset
#knn1.fit() represents  x_train is taken as X_train and y_train is taken as Y_train
knn1.fit(x_train, y_train)
# Predict the labels of the test data which means  predict the label of new data
y_test = knn1.predict(x_test)
print("---- Printing y_test----",y_test)




#----------Method 2-------------
print("------ Another method to use KNN------")
#np.array([[],[]]) represnts  a two dimensional array or matrix
X_train = np.array([[2, 2], [2, 4], [4, 5], [5, 6], [10, 6], [8, 8]])
y_train = np.array([1, 0, 1, 0, 0 , 0])
# Test data
X_test = np.array([[8, 7], [2, 5],[4, 9],[6, 11],[10, 6]])
# Initialize the model
knn = KNeighborsClassifier(n_neighbors=5)
# Fit the model to the training data
knn.fit(X_train, y_train)
# Predict the labels of the test data
y_test = knn.predict(X_test)
print("------ Pinting y_test -------")
print(y_test)

#HW2: Find the accuracy score  of this model?
#HW3: change the type of distance used  by kNN algorithm?(default is  euclidean)