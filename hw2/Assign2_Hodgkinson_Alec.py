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

np.set_printoptions(precision=4, linewidth=80, threshold=np.inf, suppress=True)


def KPCA(data, kfunc=None, num=0, alpha=0.95, spread=1):
    n, d = np.shape(data)
    if kfunc is None or kfunc.lower() == 'linear':
        K = np.dot(data, data.T)
        print(np.shape(K))
    else:
        K = np.dot(data, data.T) / (spread ** 2)
        # print(K)
        d = np.diag(K)
        K = K - np.ones(n, 1)*d.T/2
        print(d)

    # center kernel matrix
    # n = np.shape(data)[0]
    # A = np.eye(n) - np.ones((n, n))/n
    # K = np.dot(A, np.dot(K.T, A))
    #
    # # eigenvalue/vector stuff
    # nu, c = np.linalg.eigh(K)
    # nu = nu[::-1]
    # c = c[:, ::-1]
    #
    # # eigh produces a nxn matrix of eigen_vecs, but only first d are unique
    # nu = nu[:d]
    # c = c[:, :d]
    #
    # sqrt_lambda = np.sqrt(nu)
    # c = np.nan_to_num(np.divide(c, sqrt_lambda))
    #
    # lamb = nu/float(n)
    # if num == 0:
    #     total_lambda = np.sum(lamb)
    #     lambda_sum = 0.0
    #     i = 0
    #     while lambda_sum/total_lambda < alpha:
    #         lambda_sum += lamb[i]
    #         i += 1
    #     nu, c = nu[:i], c[:, :i]
    #     print('ayy"')
    #     print(np.shape(nu))
    #     print(np.shape(c))
    # else:
    #     nu, c = nu[:num], c[:, :num]
    # return np.dot(K, c)


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
    return np.dot(Z, eigen_vecs)


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
    plt.savefig("Assign2_Hodgkinson_Alec_d{0}".format(d))
    plt.show()

    keys = np.array(list(C.keys()))
    min_key = np.min(keys)
    max_key = np.max(keys)
    value_range = max_key - min_key
    mean = np.mean(angles)
    variance = np.var(angles)

    return (min_key, max_key, value_range, mean, variance)


# def test():

    ###########
    #  part 1 #
    ###########

    ###########
    #  part 2 #
    ###########
    # print(epmf())
    # print(epmf(d=100))
    # print(epmf(d=1000))

##########################################################################
##########################################################################

if __name__ == "__main__":
    # open the file with the given filename and put it in an array 1030x9 array

    with open(sys.argv[1], 'r') as f:
        d = f.read()
        data = np.array(list(map(float, re.split(r'[,\n]', d)[:-1])))
        data = data.reshape((-1, 9))
    spread = sys.argv[2]

    # print(data)
    PA = PCA(data, num=2)
    # test()
    # KA = KPCA(data, "linear", num=2)
    #
    GA = KPCA(data, 'gaussian', num=2, spread=float(spread))
    # # plt.scatter(data[:, 8], data[:, 7])
    # fig = plt.figure()
    # plt.subplot(131)
    # plt.scatter(KA[:, 0], KA[:, 1])
    # plt.xlabel("Linear KPCA")
    #
    # plt.subplot(132)
    # plt.scatter(PA[:, 0], PA[:, 1])
    # plt.xlabel("PCA")
    #
    # plt.subplot(133)
    # plt.scatter(GA[:, 0], GA[:, 1])
    # plt.xlabel("Gaussian KPCA")
    # plt.show()
