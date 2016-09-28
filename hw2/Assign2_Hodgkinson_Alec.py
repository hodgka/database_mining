# Alec Hodgkinson
# Database Mining
# 9/22/2016

####################################
#         Uses Python 3.5          #
####################################


import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import sys
import re
import scipy.stats as st

np.set_printoptions(precision=4, linewidth=80, threshold=np.inf, suppress=True)


def gaussian_kernel(data, spread):
    n, d = np.shape(data)
    K = np.zeros(shape=(n, n))

    def gaussian(x, y, spread):
        return np.exp((-(np.linalg.norm(x-y)**2))/(2*spread**2))

    X = np.asarray(data)
    i = 0
    for x in X:
        j = 0
        for y in X:
            K[i, j] = gaussian(x.T, y.T, spread)
            j += 1
        i += 1
    return K


def KPCA(data, kfunc=None, num=0, alpha=0.95, spread=1):
    n, d = np.shape(data)
    if kfunc is None or kfunc.lower() == 'linear':
        K = np.dot(data, data.T)
        print(np.shape(K))
    else:
        K = gaussian_kernel(data, spread)

    # center kernel matrix
    n = np.shape(data)[0]
    A = np.eye(n) - np.ones((n, n))/n
    K = np.dot(A, np.dot(K.T, A))

    # eigenvalue/vector stuff
    nu, eigen_vecs = np.linalg.eigh(K)
    nu = nu[::-1]
    eigen_vecs = eigen_vecs[:, ::-1]

    # eigh produces a nxn matrix of eigen_vecs, but only first d are unique
    nu = nu[:d]
    print(eigen_vecs[:, 10])
    eigen_vecs = eigen_vecs[:, :d]

    sqrt_lambda = np.sqrt(nu)
    # eigen_vecs = np.nan_to_num(np.divide(eigen_vecs, sqrt_lambda))

    lamb = nu/float(n)
    if num == 0:
        total_lambda = np.sum(lamb)
        lambda_sum = 0.0
        i = 0
        while lambda_sum/total_lambda < alpha:
            lambda_sum += lamb[i]
            i += 1
        nu, eigen_vecs = nu[:i], eigen_vecs[:, :i]
    else:
        nu, eigen_vecs = nu[:num], eigen_vecs[:, :num]
    print("KERNEL: ", eigen_vecs)
    return np.dot(K, eigen_vecs)


def PCA(data, num=0, alpha=0.95):
    # center the data
    mean = np.mean(data, axis=0)
    Z = data - mean

    # calculate covariance matrix and eigen values is descending order
    # 9x9 matrix
    Sigma = np.cov(Z.T)
    eigen_vals, eigen_vecs = np.linalg.eigh(Sigma)
    eigen_vals = eigen_vals[::-1]
    eigen_vecs = eigen_vecs[:, ::-1]

    # use alpha to compute fraction of total variance
    if num == 0:
        total_lambda = np.sum(eigen_vals)
        i = 0
        lambda_sum = 0.0
        while lambda_sum/total_lambda < alpha:
            lambda_sum += eigen_vals[i]
            i += 1
        eigen_vecs = eigen_vecs[:, :i]
    else:
        eigen_vecs = eigen_vecs[:, :num]
    print("PCA: ", eigen_vecs)
    return np.dot(Z, eigen_vecs)


##########################################################################
##########################################################################


def epmf(n=100000, d=10):

    half_diagonals = np.array(list(itertools.product([1, -1], repeat=d)))
    # randomly choose 2n half diagonals
    diag_inds = np.random.choice(np.shape(half_diagonals)[0], 2*n)
    # make tuples of the n pairs of diagonals
    diags = half_diagonals[diag_inds, :]
    diags = zip(*[iter(diags)]*2)
    # calculate the angle

    def angle(x1, x2):
        return np.degrees(np.arccos(np.dot(x1, x2) /
                          (np.linalg.norm(x1) * np.linalg.norm(x2))))

    angles = [angle(vec[0], vec[1]) for vec in diags]

    C = Counter(angles)
    total = sum(C.values())
    for key, value in C.items():
        C[key] = value / float(total)
    plt.bar(C.keys(), C.values(), color='g')
    plt.xlabel("angle(degrees)")
    plt.ylabel("percent of diagonals")
    plt.savefig("Assign2_Hodgkinson_Alec_P2_d{0}".format(d))
    plt.show()

    keys = np.array(list(C.keys()))
    min_key = np.min(keys)
    max_key = np.max(keys)
    value_range = max_key - min_key
    mean = np.mean(angles)
    variance = np.var(angles)

    return (min_key, max_key, value_range, mean, variance)

##########################################################################
##########################################################################

if __name__ == "__main__":
    # open the file with the given filename and put it in an array 1030x9 array

    with open(sys.argv[1], 'r') as f:
        d = f.read()
        data = np.array(list(map(float, re.split(r'[,\n]', d)[:-1])))
        data = data.reshape((-1, 9))

    spread1 = sys.argv[2]

    KA = KPCA(data, "linear", num=2)
    PA = PCA(data, num=2)
    GA1 = KPCA(data, 'gaussian', num=2, spread=float(spread1))

    fig = plt.figure()
    plt.subplot(131)
    plt.scatter(KA[:, 0], KA[:, 1])
    plt.xlabel("Linear KPCA")

    plt.subplot(132)
    plt.scatter(PA[:, 0], PA[:, 1])
    plt.xlabel("PCA")

    plt.subplot(133)
    plt.scatter(GA1[:, 0], GA1[:, 1])
    plt.xlabel("Gaussian KPCA")
    plt.savefig("Assign2_Hodgkinson_Alec_P1")
    plt.show()

    print(epmf(d=10))
    print(epmf(d=100))
    print(epmf(d=1000))
