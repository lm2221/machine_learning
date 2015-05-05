import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback

from datetime import datetime

# relevant files:
# cancer_csv
# X.csv, y.csv

X = pd.read_csv("cancer_csv/X.csv", header=None)
Y = pd.read_csv("cancer_csv/y.csv", header=None)

Xtrain = X[183:]
Ytrain = Y[183:]
Xtest = X[:183]
Ytest = Y[:183]


def sample_w(w, n):
    c = np.random.random(n)
    for i in xrange(n):
        cdf_value = 0
        for j,p in enumerate(w):
            cdf_value += p
            if c[i] < cdf_value:
                c[i] = j
                break
    return c


def bootstrap_sample(X, Y, w):
    N,D = X.shape
    X = X.as_matrix()
    Y = Y.as_matrix()
    C = sample_w(w, N)
    Xb = np.zeros(X.shape)
    Yb = np.zeros(N)
    for i,j in enumerate(C):
        Xb[i] = X[j]
        Yb[i] = Y[j]
    return pd.DataFrame(Xb), pd.DataFrame(Yb)


class Bayes(object):
    def fit(self, X, Y):
        N, D = X.shape
        
        cov = np.cov(X[range(1,D)].T)
        icov = np.linalg.inv(cov)
        
        mu1 = X[Y[0] == 1][range(1,D)].mean()
        mu0 = X[Y[0] == -1][range(1,D)].mean()
        pi1 = len(Y[Y[0] == 1]) / float(N)
        pi0 = len(Y[Y[0] == -1]) / float(N)

        if pi1 == 0:
            raise Exception("no values for class 1")
        if pi0 == 0:
            raise Exception("no values for class 0")

        w0 = np.log(pi1/pi0) - 0.5*np.dot( np.dot(mu1 + mu0, icov), mu1 - mu0 )
        w = np.dot(icov, mu1 - mu0)
        self.w = np.append(w0, w)


    def predict(self, X):
        return np.sign(np.dot(X, self.w))


    def accuracy(self, X, Y):
        pred = self.predict(X)
        return sum(np.abs(pred == Y.as_matrix().flatten())) / float(len(Y))


class BinaryLogistic(object):
    # expect input Y to be -1/1
    # but train based on Y in 0/1

    def fit(self, X, Y):
        # make it easy to switch between online and full
        #self.fit_full(X, Y)
        self.fit_online(X, Y)


    def fit_online(self, X, Y, lr=0.01, l2_regularization=1, early_stop=False):
        # Note: if you set lr too low (i.e. 0.00001), error will not go below 0.5
        N,D = X.shape
        target = Y.apply(lambda y: 1 if y[0] == 1 else 0, axis=1) # train with 0-1
        self.w = np.zeros(D)
        accuracies = np.zeros(N)
        indices = range(N)
        allWs = np.zeros((N,D))
        np.random.shuffle(indices)
        for i in indices:
            allWs[i] = self.w
            x = X.loc[i]
            t = target.loc[i]

            p = self.forward(x)
            # print "target:", t, "probability:", p
            accuracies[i] = self.accuracy(X, Y)

            if accuracies[-1] > 0.5 and early_stop:
                break
            self.w += lr*((t - p)*x - l2_regularization*self.w)

        # sometimes after training classifier is worse than chance
        # if so, train again but stop after training error is < 0.5
        if accuracies[-1] < 0.5:
            print "final online accuracy:", accuracies[-1]
            # first see if we can look at the past accuracies and choose the best one
            best_idx = np.argmax(accuracies)
            if accuracies[best_idx] > 0.5:
                self.w = allWs[best_idx]
            elif not early_stop:
                print "try again with early stop"
                # plt.plot(accuracies)
                # plt.title("Logistic online learning accuracies vs epochs")
                # plt.show()
                self.fit_online(X, Y, lr, l2_regularization, True)
            else:
                print "used early stop and still could not do better than 0.5 accuracy"


    def fit_full(self, X, Y, lr=0.00001, l2_regularization=1, epochs=1000):
        N,D = X.shape
        target = Y.apply(lambda y: 1 if y[0] == 1 else 0, axis=1) # train with 0-1
        self.w = np.zeros(D)
        accuracies = np.zeros(epochs)
        for e in xrange(epochs):
            probs = self.forward(X)
            accuracies[e] = self.accuracy(X, Y)
            self.w += lr*(np.dot((target - probs).T, X) - l2_regularization*self.w)

    def forward(self, X):
        return 1 / ( 1 + np.exp( -np.dot(X, self.w) ) )


    def predict(self, X):
        P = self.forward(X)
        N = len(X)
        Y = np.zeros(N)
        for i in xrange(N):
            if P[i] >= 0.5:
                Y[i] = 1
            else:
                Y[i] = -1
        return Y

    def accuracy(self, X, Y):
        pred = self.predict(X)
        return sum(np.abs(pred == Y.as_matrix().flatten())) / float(len(Y))


class AdaBoost(object):
    def __init__(self, Classifier):
        self.create_classifier = Classifier


    # only pass in Xtest and Ytest to show training error and test error graphs
    def fit(self, X, Y, Xtest, Ytest, T=1000):
        N = len(X)
        W = np.ones(N)/N
        Yflat = Y.as_matrix().flatten()
        self.classifiers = []
        self.alphas = np.zeros(T)

        # save these to answer questions
        training_errors = np.zeros(T)
        testing_errors = np.zeros(T)
        individual_test_errors = np.zeros(T)
        epsilons = np.zeros(T)
        WT = np.zeros((T, N))
        

        for t in xrange(T):
            # keep running because clf may throw exception
            # due to no samples in one class, singular covariance, etc.
            for i in xrange(5):
                try:
                    Xb, Yb = bootstrap_sample(X, Y, W)
                    clf = self.create_classifier()
                    clf.fit(Xb, Yb)
                    break
                except Exception as e:
                    print e
            self.classifiers.append(clf)

            # save for the homework
            e_train = self.error_rate(X, Y)
            e_test = self.error_rate(Xtest, Ytest)
            training_errors[t] = e_train
            testing_errors[t] = e_test
            individual_test_errors[t] = clf.accuracy(Xtest, Ytest)

            y_pred = clf.predict(X)
            epsilon = sum(wi for wi,p,y in zip(W, y_pred, Y) if p != y)
            epsilons[t] = epsilon
            alpha = 0.5*np.log( (1 - epsilon) / epsilon )
            self.alphas[t] = alpha

            # update W
            tmp = W*np.exp(-alpha * Yflat * y_pred) + 0.0001 # smooth it out
            W = tmp / sum(tmp)

            WT[t] = W

            # each iteration goes slower than the last
            if t > 20:
                print "t:", t

        print "final test accuracy: %.4f" % (1 - e_test)

        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # plot training and test errors vs t
        plt.plot(training_errors, label='Training errors')
        plt.plot(testing_errors, label='Testing errors')
        plt.legend(["Training errors", "Testing errors"])
        plt.title("Training and test errors")
        plt.savefig("%s_train_and_test_errors.png" % dt)

        # plot test error for current bayes model
        plt.figure()
        plt.plot(individual_test_errors, label="Individual test errors")
        plt.title("Individual test errors")
        plt.savefig("%s_individual_test_errors.png" % dt)

        # plot alpha vs t
        plt.figure()
        plt.plot(self.alphas)
        plt.title("Alpha vs t")
        plt.savefig("%s_alphas.png" % dt)

        # plot epsilon vs t
        plt.figure()
        plt.plot(epsilons)
        plt.title("Epsilon vs t")
        plt.savefig("%s_epsilons.png" % dt)

        # plot epsilon vs alpha
        plt.figure()
        plt.scatter(epsilons, self.alphas)
        plt.title("Epsilon vs alpha")
        plt.savefig("%s_epsilon_vs_alpha.png" % dt)

        # plot W[i] for i=100,200,300
        plt.figure()
        plt.plot(WT)
        plt.title("W vs t")
        plt.savefig("%s_weights.png" % dt)

        plt.figure()
        plt.plot(WT.T[50])
        plt.plot(WT.T[75])
        plt.plot(WT.T[100])
        plt.legend(["50","75","100"])
        plt.title("W[50,75,100] vs t")
        plt.savefig("%s_weights[50,75,100].png" % dt)


    def predict(self, X):
        y = np.zeros(len(X))
        for t, clf in enumerate(self.classifiers):
            y += self.alphas[t] * clf.predict(X)
        return np.sign(y)


    def error_rate(self, X, Y):
        pred = self.predict(X)
        return sum(np.abs(pred != Y.as_matrix().flatten())) / float(len(Y))


if __name__ == '__main__':
    # part 1
    if '1' in sys.argv:
        w = [0.1, 0.2, 0.3, 0.4]
        for n in (100, 200, 300, 400, 500):
            c = sample_w(w, n)
            plt.hist(c, bins=4)
            plt.title("n = %d" % n)
            plt.show()

    # part 2
    # test accuracy without boosting
    if '2' in sys.argv:
        bayes = Bayes()
        bayes.fit(Xtrain, Ytrain)
        print "accuracy of Bayes without boosting: %.4f" % bayes.accuracy(Xtest, Ytest)

        adaboost = AdaBoost(Bayes)
        adaboost.fit(Xtrain, Ytrain, Xtest, Ytest)

    # part 3
    # boosting with logistic regression
    if '3' in sys.argv:
        lr = BinaryLogistic()
        lr.fit_full(Xtrain, Ytrain)
        print "accuracy of Logistic Regression without boosting: %.4f" % lr.accuracy(Xtest, Ytest)

        adaboost = AdaBoost(BinaryLogistic)
        adaboost.fit(Xtrain, Ytrain, Xtest, Ytest)
