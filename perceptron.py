# Perceptron
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import pandas as pd

iris = datasets.load_iris()
print(dir(iris))
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df)

x = iris.data[:, :2]
y = (iris.target == 0).astype(int)

print(df.dtypes)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Y_test")
print(y_train)

perceptron = Perceptron(max_iter=100, tol=1e-3)

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=y, cmap="Paired")
plt.show()
score = perceptron.score(X_test, y_test)
print("Accuray", score)

