"""
This file will become at some point a test for different proposals, comparing:
- speed
- spread of the points in #sigmas distance
- ...
"""

from gpry.proposal import Centroids
import scipy.stats as stats
import numpy as np
from time import time
import matplotlib.pyplot as plt

# Dimensionality, number of traning points (drawn from gaussian), and of proposed points
d = 10
n_train = 300
n_proposed = 100

X = stats.multivariate_normal.rvs(mean=np.zeros(d), cov=np.eye(d), size=n_train)

# Lambda parameter of the exponential.
# smaller = more spread.
# adapt to dimensionality, softly: e.g. 1 for d=2 and 0.5 for d=30
expon_lambda = 1

prop = Centroids(d * [[0, 1]], X, lambd=expon_lambda)

start = time()
proposed = np.array([prop.get() for _ in range(n_proposed)])
total_time = time() - start
print(f"Proposal generation took {total_time} sec for {n_proposed} vectors.")

# Dimensions to plot
d_i, d_j = 0, 1

plt.figure(figsize=(10, 8), facecolor="w")
plt.scatter(*X[:, (d_i, d_j)].T, label="training")
plt.scatter(*proposed[:, (d_i, d_j)].T, marker="+", label="proposed")
plt.legend()
plt.show()
