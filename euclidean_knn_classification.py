#The Euclidean Distance


import math
print("Enter the first point A")
x1, y1  = map(int, input().split())
print("Enter the second point B")
x2, y2 = map(int, input().split())
dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
print("The Euclidean Distance is " + str(dist))


# knn

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
print(y_test)
