import numpy as np 

class KMeans:
    def __init__(self, K=2, max_iter=100): 
        self.K = K 
        self.max_iter = max_iter

    def fit(self, X):
        N, D = X.shape 
        print(X.shape[1]) 
        print(N) 
        print(D) 
        self.centroids = X[np.random.choice(N, self.K, replace=False), :]
        for i in range(self.max_iter): 
            distances = np.zeros((N, self.K)) 
            print(distances) 
            for j in range(self.K):
                distances[:, j] = np.sum((X - self.centroids[j, :]) ** 2, axis=1) 
            cluster_assignments = np.argmin(distances, axis=1) 
            for j in range(self.K): 
                self.centroids[j, :] = np.mean(X[cluster_assignments == j, :], axis=1) 
                self.cluster_assignments = cluster_assignments 
                return self

def predict(self, X): 
    N, D = X.shape
    distances = np.zeros((N, self.K)) 
    for j in range(self.K): 
        distances[:, j] = np.sum((X - self.centroids[j, :]) ** 2, axis=1) 
    return np.argmin(distances, axis=1)

X = np.array([[1, 2], [3, 4], [5, 6]]) 
kmeans = KMeans(K=2).fit(X) 
predictions = kmeans.predict(X) 
print(predictions)
