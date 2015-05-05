import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.cluster import KMeans as KMeansSKL


class KMeans(object):
    def __init__(self, k, max_iter=20):
        self.k = k
        self.max_iter = 20


    def fit(self, X):
        N = len(X)
        initial_indices = np.random.choice(np.arange(N), self.k)
        self.centers = np.array([X[i] for i in initial_indices])

        L = np.zeros(self.max_iter)
        for i in xrange(self.max_iter):
            # 1. determing current cluster assignments
            Y = self.predict(X)

            # 2. find new cluster centers
            self.get_centers(X,Y)

            L[i] = self.objective(X,Y)
        return L


    def predict(self, X):
        # return index of center
        y = np.zeros(len(X), dtype=int)
        for i,x in enumerate(X):
            min_c = -1
            min_dist = float("inf")
            for c,m in enumerate(self.centers):
                diff = x - m
                d = np.dot(diff, diff)
                if d < min_dist:
                    min_c = c
                    min_dist = d
            y[i] = min_c
        return y


    def get_centers(self, X, Y):
        self.centers = np.array([X[Y == k].mean(axis=0) for k in xrange(self.k)])


    def objective(self, X, Y):
        return sum(np.dot(x - self.centers[k], x - self.centers[k]) for k in xrange(self.k) for x in X[Y == k])
        
# k-means on Gaussian dataset
N = 500
priors = [0.2, 0.5, 0.3]
mus = [
    np.array([0, 0]),
    np.array([3, 0]),
    np.array([0, 3]),
]
X = np.zeros((N, 2))
for i in xrange(N):
    j = choose_center(priors)
    X[i] = np.random.randn(2) + mus[j]

for K in (2,3,4,5):
    kmeans = KMeans(K)
    L = kmeans.fit(X)

    # plot value of k-means objective function
    plt.plot(L)
    plt.title("Objective function values for k = %s" % K)
    plt.show()

    # scatter plots
    if K in (3,5):
        y = kmeans.predict(X)
        plt.scatter(X[:,0], X[:,1], s=100, c=y, alpha=0.5)
        plt.title("Cluster assignments for k = %s" % K)
        plt.show()