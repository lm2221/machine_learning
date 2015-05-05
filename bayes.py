import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# relevant files:
# mnist_csv
# label_text.txt, label_train.txt, Q.txt, Xtest.txt, Xtrain.txt

# easier to work with as pandas dataframes because we can filter classes
Xtest = pd.read_csv("mnist_csv/Xtest.txt", header=None)
Xtrain = pd.read_csv("mnist_csv/Xtrain.txt", header=None)
Ytest = pd.read_csv("mnist_csv/label_test.txt", header=None)
Ytrain = pd.read_csv("mnist_csv/label_train.txt", header=None)

class Bayes(object):
    def fit(self, X, y):
        self.gaussians = dict()
        labels = set(y.as_matrix().flatten())
        for c in labels:
            current_x = X[y[0] == c]
            self.gaussians[c] = {
                'mu': current_x.mean(),
                'sigma': current_x.var().as_matrix() * np.eye(X.shape[1]),
            }

    def predict_one(self, x):
        max_ll = float("-inf")
        max_c = -1 # sentinel value
        for c,g in self.gaussians.iteritems():
            x_minus_mu = x - g['mu']
            k1 = np.log(2*np.pi*np.linalg.det(g['sigma']))
            k2 = np.dot( np.dot(x_minus_mu, np.linalg.inv(g['sigma'])), x_minus_mu )
            ll = -0.5*(k1 + k2)

            if ll > max_ll:
                max_c = c
                max_ll = ll
        return max_c

    def predict(self, X):
        Ypred = X.apply(self.predict_one, axis=1)
        return Ypred

    def distributions(self, x):
        lls = np.zeros(len(self.gaussians))
        for c,g in self.gaussians.iteritems():
            x_minus_mu = x - g['mu']
            k1 = np.log(2*np.pi*np.linalg.det(g['sigma']))
            k2 = np.dot( np.dot(x_minus_mu, np.linalg.inv(g['sigma'])), x_minus_mu )
            ll = -0.5*(k1 + k2)
            lls[c] = ll
        return lls


# show confusion matrix and accuracy number
bayes = Bayes()
bayes.fit(Xtrain, Ytrain)
Ypred = bayes.predict(Xtest)
C = np.zeros((10,10), dtype=np.int)
print len(Ypred), len(Ytest)
for p,t in zip(Ypred.as_matrix().flatten(), Ytest.as_matrix().flatten()):
    C[t,p] += 1
print "Confusion matrix:"
print C
print "Accuracy:", np.trace(C) / 500.0

# show means as images
Q = pd.read_csv("mnist_csv/Q.txt", header=None).as_matrix()
for c,g in bayes.gaussians.iteritems():
    y = np.dot(Q, g['mu'].as_matrix())
    y = np.reshape(y, (28,28))
    plt.imshow(y)
    plt.title(c)
    plt.show()

# show distributions for 3 misclassified examples
print "distributions for 3 misclassified examples:"
count = 0
for i,p in Ypred.iteritems():
    if p != Ytest.loc[i][0]:
        print "predicted:", p, "actual:", Ytest.loc[i][0]
        print bayes.distributions(Xtest.loc[i])
        count += 1
    if count >= 3:
        break