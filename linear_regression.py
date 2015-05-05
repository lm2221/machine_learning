import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# relevant files:
# data_csv
# legend.txt, X.txt, y.txt

X = pd.read_csv("X.txt", header=None)
y = pd.read_csv("y.txt", header=None)
lr = LinearRegression()

def split_train_predict(X, y, show=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20)

    m = np.matrix(X_train)
    # use built in function
    w = np.linalg.lstsq(X_train, y_train)[0]
    y_pred = X_test*np.matrix(w)
    
    # print the values of w
    if show:
        print "w:", w
    return y_test - y_pred

maes = np.zeros(1000)
diffs = np.zeros((4, 20*1000))
rmses = np.zeros((4, 1000))
for i in xrange(1000):
    diff = split_train_predict(X, y, i == 0)
    maes[i] = np.mean(abs(diff))
    diffs[0, i*20:(i+1)*20] = diff.T
    rmses[0, i] = np.sqrt( np.mean( np.multiply(diff, diff) ) )

print "mean MAE:", np.mean(maes), "std MAE:", np.std(maes)
print "mean RMSE:", np.mean(rmses[0]), "std RMSE:", np.std(rmses[0])
plt.hist(diffs[0], bins=50)
plt.show()

# extend X
D = 7
for p in (2,3,4):
    # add x(i)^p
    for d in xrange(D):
        X["%d_%d" % (d+1,p)] = X[d]**p

    for i in xrange(1000):
        diff = split_train_predict(X, y)
        diffs[p-1, i*20:(i+1)*20] = diff.T
        rmses[p-1, i] = np.sqrt( np.mean( np.multiply(diff, diff) ) )

    print "mean RMSE:", np.mean(rmses[p-1]), "std RMSE:", np.std(rmses[p-1])
    plt.hist(diffs[p-1], bins=50)
    plt.show()

# plot log-likelihoods of errors vs p
ll_vs_p = np.zeros(4)
for p in xrange(4):
    m = np.mean(diffs[p])
    s = np.std(diffs[p])
    ll = sp.stats.norm.logpdf(diffs[p], m, s)
    ll_vs_p[p] = np.sum(ll)
plt.plot([1,2,3,4], ll_vs_p)
plt.show()