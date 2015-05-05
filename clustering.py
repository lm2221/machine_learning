import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.cluster import KMeans as KMeansSKL

# relevant files:
# movies_csv
# movies.txt, ratings_test.txt, ratings.txt

class MatrixFactorization(object):
    def fit(self, Rtrain, Rtest, N1, N2, d=20, sigma_sq=0.25, reg=10, max_iter=100):
        # assume user ids = 1...N1
        # assume movie ids = 1...N2
        # return RMSE and log-likelihood at each iteration
        self.U = np.random.random((N1, d))
        self.V = np.random.random((N2, d))

        rmse = np.zeros(max_iter)
        const = -0.5*np.log(2*np.pi*sigma_sq)*len(Rtest) + 2*d*np.log(reg/(2*np.pi))*(N1 + N2)
        ll = np.ones(max_iter)*const
        c1 = 1/(2*sigma_sq)
        c2 = reg/2
        lambda_sigma_sq_I = reg*sigma_sq*np.eye(d)

        for t in xrange(max_iter):
            # calculate rmse and log-likelihood
            current_rmse = 0
            for i,j,r in Rtest:
                p = self.predict(i,j)
                ll[t] += -c1*(r-p)*(r-p)
                rounded_diff = r - self.__class__.round_prediction(p)
                current_rmse += rounded_diff*rounded_diff
            for i in xrange(N1):
                ll[t] -= c2*np.dot(self.U[i], self.U[i])
            for j in xrange(N2):
                ll[t] -= c2*np.dot(self.V[j], self.V[j])
            rmse[t] = np.sqrt(current_rmse/len(Rtest))

            # updates for u and v
            u_outer = np.zeros((N1,d,d)) # a dxd matrix for each of the N1 users
            v_outer = np.zeros((N2,d,d)) # a dxd matrix for each of the N2 movies
            u_vector = np.zeros((N1,d))
            v_vector = np.zeros((N2,d))
            for i,j,r in Rtrain:
                u_outer[i] += np.outer(self.V[j], self.V[j])
                u_vector[i] += r*self.V[j]
                v_outer[j] += np.outer(self.U[i], self.U[i])
                v_vector[j] += r*self.U[i]
            for i in xrange(N1):
                try:
                    self.U[i] = np.dot(np.linalg.inv(lambda_sigma_sq_I + u_outer[i]), u_vector[i])
                except Exception as e:
                    print "u_outer[%d]: %s" % (i, u_outer[i])
                    raise e
            for j in xrange(N2):
                self.V[j] = np.dot(np.linalg.inv(lambda_sigma_sq_I + v_outer[j]), v_vector[j])
            print "Finished iteration:", t

        return rmse, ll

    def predict(self, i, j):
        return np.dot(self.U[i], self.V[j])

    @classmethod
    def round_prediction(cls, x):
        if x > 5:
            return 5
        elif x < 1:
            return 1
        else:
            return round(x)


def choose_center(priors):
    # choose which center to sample from
    p = np.random.random()
    cdf = 0
    for j, prior in enumerate(priors):
        cdf += prior
        if p < cdf:
            return j


def find_close_movies(V, titles):
    # V is a N2 x d matrix
    distance_title = []
    for i in (0,1,6):
        # query_index = np.random.randint(len(V))
        query = V[i]
        for j,v in enumerate(V):
            diff = query - v
            dist = np.dot(diff, diff)
            if j != i:
                distance_title.append((dist, titles[j]))
        ordered = sorted(distance_title)

        print "QUERY:", titles[i]
        for _, title in ordered[:5]:
            # print the top 5
            print "\t", title


def find_movies_close_to_user_archetype(mf, titles):
    kmeans = KMeans(30)
    #kmeans = KMeansSKL(30)
    L = kmeans.fit(mf.U)
    plt.plot(L)
    plt.title("KMeans 30")
    plt.show()
    center_indices = np.random.choice(range(30), 5, replace=False)
    for i in center_indices:
        u = kmeans.centers[i]
        #u = kmeans.cluster_centers_[i]
        ratings = np.dot(u, mf.V.T)
        ind = np.argpartition(ratings, -10)[-10:]
        print "MOVIE GROUP:", u
        for j in ind:
            print titles[j], ratings[j]
        max_rating_index = np.argmax(ratings)
        print "closest movie:", titles[max_rating_index]
        print ""

# part 2: matrix factorization on movie data
# NOTE: movie ids start from 1
Rtrain = pd.read_csv('movies_csv/ratings.txt', header=None)
Rtest = pd.read_csv('movies_csv/ratings_test.txt', header=None)
N1 = max(Rtrain[0].max(), Rtest[0].max())
N2 = max(Rtrain[1].max(), Rtest[1].max())

# make the indexes/ids go from 0...N-1
Rtrain[0] = Rtrain[0] - 1
Rtrain[1] = Rtrain[1] - 1
Rtest[0] = Rtest[0] - 1
Rtest[1] = Rtest[1] - 1

mf = MatrixFactorization()
rmse, ll = mf.fit(Rtrain.as_matrix(), Rtest.as_matrix(), N1, N2, max_iter=100)

# plot RMSE per iteration
plt.plot(rmse)
plt.title("RMSE vs t")
plt.show()

# plot loglikelihood
plt.plot(ll)
plt.title("log-likelihood vs t")
plt.show()

# print closet movie to the query
titles = []
for line in open("movies_csv/movies.txt"):
    titles.append(line.rstrip())
find_close_movies(mf.V, titles)

# find closest movie to the user archetype
find_movies_close_to_user_archetype(mf, titles)
