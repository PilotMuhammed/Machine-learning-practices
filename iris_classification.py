from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

iris = datasets. load_iris()
x,y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
print(x_train)

from matplotlib.colors import ListedColormap
Colormap = ListedColormap (['b','r','g'])
plt.figure()
plt.scatter (x [:,0], x[:,1], c=y, cmap = Colormap)
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x, y)
pred = knn. predict (x_test)
print (pred)
def accu(y_test, pred):
    acc = np.sum (y_test == pred) / len (y_test)
    return acc
print ('Accurcy:', accu(y_test, pred))

