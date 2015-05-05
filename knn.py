import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

X_train=pd.read_csv("mnist_csv/Xtrain.txt",header=None).as_matrix()
X_test=pd.read_csv("mnist_csv/Xtest.txt",header=None).as_matrix()
y_train=pd.read_csv("mnist_csv/label_train.txt",header=None).as_matrix().flatten()
y_test=pd.read_csv("mnist_csv/label_test.txt",header=None).as_matrix().flatten()

class KNN(object):
    def __init__(self,k):
        self.k = k
    
    def fit(self, X, Y):
        self.Xtrain = X
        self.Ytrain = Y

    def predict(self, X):
        ypred = np.zeros(len(X))
        for i,x in enumerate(X):
            distances=[]
            # min_dist=float("inf")
            for x_train, y_train in zip(self.Xtrain,self.Ytrain):
                dx = x - x_train
                dist = np.sum(dx*dx)

                if len(distances)<self.k:
                    distances.append( (dist, y_train) )
                else:
                    # print "distances:", distances
                    max_idx = np.argmax(distances, axis=0)[0]
                    # print "dist:", dist, "max_dist:", distances[max_idx]
                    if dist < distances[max_idx][0]:
                        del distances[max_idx]
                        distances.append( (dist, y_train) )
                ypred[i] = self.vote(distances)
        return ypred

#pick the highest number that appears
    def vote(self, distances):
        count=dict()
        max_votes = -1
        max_y = -1
        for x,y in distances:
            count[y] = count.get(y,0) + 1
            if count[y] > max_votes:
                max_votes = count[y]
                max_y = y
        return max_y

for k in (1,2,3,4,5):
    knn = KNN(k)
    knn.fit(X_train,y_train)
    ypred=knn.predict(X_test)

#show 3 miss classified predictions
    C=np.zeros((10,10))
    count_errors_printed = 0
    for t,p in zip(y_test, ypred):
        C[t,p] += 1
        if t != p and count_errors_printed < 3:
            count_errors_printed += 1
            print "target:", t, "prediction:", p

#show confusion matrix and accuracy
    print C
    print "accuracy:", np.trace(C)/500.0

