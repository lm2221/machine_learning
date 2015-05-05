import numpy as np
import matplotlib.pyplot as plt

# relevant files:
# hw5text
# cfb2014scores.csv, legend.txt

# given: number of teams = 759
team_names = []
for line in open('hw5text/legend.txt'):
    team_names.append(line.rstrip())

D=759
M = np.zeros((D,D))
for line in open('hw5text/cfb2014scores.csv'):
    r = line.split(',')
    i, score_i, j, score_j = [int(x) for x in r]

# index by 0 in Python
    i -= 1
    j -= 1

    if score_i > score_j:
    # team 1 wins

        M[i,i] += 1
        M[j,i] += 1

    elif score_j > score_i:
    # team 2 wins

        M[j,j] += 1
        M[i,j] += 1


    proportion_i = float(score_i) / (score_i + score_j)
    proportion_j = float(score_j) / (score_i + score_j)

    M[i,i] += proportion_i
    M[j,j] += proportion_j
    M[i,j] += proportion_j
    M[j,i] += proportion_i

print M
 # normalize M --> question how do we know not to normalize over y?
for i in xrange(D):
    M[i] /= np.sum(M[i])

w0 = np.ones(D)/D # uniform

lam, v = np.linalg.eig(M.T)
u = v[:, lam.argmax()]
u_normalized = u/u.sum()

T = 1000
w = w0
w_minus_u = np.zeros(T)
for t in xrange(T):
    w = np.dot(w,M)
    #w_minus_u[t] = np.sum(np.abs(w - u_normalized))
    w_minus_u[t] = np.linalg.norm(w-u_normalized,1)
    if t==1000:
        print "t", t+1, w_minus_u[t]
    #print w_minus_u[1000]

    if t+1 in (10, 100, 200, 1000):
        # top 20 teams with corresponding w values
        print "Top 20 teams for t =", t+1
        sorted_indices = (-w).argsort()
        for j in xrange(20):
            print team_names[sorted_indices[j]], w[sorted_indices[j]]
        print ""

print "final objective value:", w_minus_u[T-1]

plt.title("w_minus_u for T=1000")
plt.plot(w_minus_u)
plt.show()

