# Alec Hodgkinson
# Database Mining
# 9/22/2016

####################################
#         Uses Python 3.5          #
####################################

import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import sys
import re
from sklearn.decomposition import PCA, KernelPCA

np.set_printoptions(precision=4, linewidth=80, suppress=True)

##########################################################################
##########################################################################
# Takes 3 arguments with flags -fname, -func, and -eps(optional).
# The arguments are the filename to parse data from, the function you want to
# execute, and the value of epsilon you want to use.
##########################################################################
##########################################################################
# Possible function arguments include: mean, variance, "covariance inner",
# "covariance outer", correlation, eigenvector, and all
##########################################################################
##########################################################################
# Epsilon value should be given in the format " 10e-10"
##########################################################################
##########################################################################
# parser = argparse.ArgumentParser(description="Database Mining Homework 2")
# parser.add_argument('-fname', metavar='filename', type=str,
#                     default='airfoil_self_noise.txt',
#                     help='File to collect data from.')
# parser.add_argument('-func', metavar="function", type=str, default='mean',
#                     help="Function to call.")
# parser.add_argument('-eps', metavar="epsilon", type=float,
#                     help="Value for epsilon.")
# args = parser.parse_args()


def create_kernel(data, kfunc=None, spread=1):
    assert(type(kfunc) == str)
    if kfunc is None or kfunc.lower() == 'linear':
        K = np.dot(data, data.T)
    else:
        K = np.exp(np)

    n = np.shape(data)[0]
    A = np.eye(n) - np.ones((n, n))/n
    K = np.dot(A, np.dot(K.T, A))

    return K

# def normalize_kernel(K):
    w = np.diag(1./np.sqrt(np.diag(K)))
    K_normalized = np.dot(w, np.dot(K.T, w))
    return K_normalized


def dominant_eigenvectors(K, n=0, alpha=None):
    eigen_val, eigen_vec = np.linalg.eigh(K)
    eigen_val = eigen_val[::-1]
    eigen_vec = eigen_vec[:, ::-1]
    lambda_i = eigen_val/float(np.shape(K)[0])
    # TODO need to scale eigen vectors and finish KPCA
    # eigen_vec = np.nan_to_num(np.divide(eigen_vec, inv_sqrt_lambda))
    if n != 0:
        # if n is specified, return n dominant dimensions
        print(eigen_vec)
        return (eigen_val[:n], eigen_vec[:, :n])
    elif alpha is not None:
        # if alpha is specified, then find the r dominant dimensions
        total_lambda = np.sum(eigen_val)
        lambda_sum = 0
        i = 0
        while float(lambda_sum)/total_lambda < alpha:
            lambda_sum += eigen_val[i]
            i += 1
        print(eigen_vec)
        return (eigen_val[:i], eigen_vec[:, :i])
    else:
        return(eigen_val[:], eigen_vec[:, :])


def reduce_data_dimensionality(K, C):
    return np.dot(C.T, K)


def create_covariance(data):
    return np.cov(data.T)


def dominant_pca_eigenvectors(S, n=0, alpha=None):
    eigen_val, eigen_vec = np.linalg.eigh(S)
    eigen_val = eigen_val[::-1]
    eigen_vec = eigen_vec[:, ::-1]
    if n != 0:
        # if n is specified, return n dominant dimensions
        print(eigen_vec)
        return (eigen_val[:n], eigen_vec[:, :n])
    elif alpha is not None:
        # if alpha is specified, then find the r dominant dimensions
        total_lambda = np.sum(eigen_val)
        lambda_sum = 0
        i = 0
        while float(lambda_sum)/total_lambda < alpha:
            lambda_sum += eigen_val[i]
            i += 1
        print(eigen_vec)
        return (eigen_val[:i], eigen_vec[:, :i])
    else:
        print("Alpha is None")


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
    plt.show()

    keys = np.array(list(C.keys()))
    min_key = np.min(keys)
    max_key = np.max(keys)
    value_range = max_key - min_key
    mean = np.mean(angles)
    variance = np.var(angles)

    return (min_key, max_key, value_range, mean, variance)



def test():

    ###########
    #  part 1 #
    ###########

    kernel_data = np.array([[540, 162, 2.5], [540, 162, 2.5], [332.5, 228, 0]])
    # K = create_kernel(kernel_data, 'linear')
    # K = center_kernel(K)
    # eigen_vals, eigen_vecs = dominant_eigenvectors(K)
    # print(eigen_vals)
    # print(eigen_vecs)
    # print(K)

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

    test()
    # K = create_kernel(data, "linear")
    # K_normalized = normalize_kernel(K)
    # C = dominant_eigenvectors(K_normalized, n=2, alpha=0.95)[1]
    # # print(C)
    # A = reduce_data_dimensionality(K, C)
    # print(np.shape(K))
    # print((C))
    # # for i in range(9):
    # #     for i in range():
    # #         plt.scatter(data[:, i], data[:, j])
    # # plt.scatter(data[:, 8], data[:, 7])
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.scatter(A[0], A[1])
    # # plt.show()
    # # variance = float(sys.argv[2])
    # S = create_covariance(data)
    # C_pca = dominant_pca_eigenvectors(S, 2)[1]
    # A_pca = np.dot(data, C_pca)
    # print(np.shape(A_pca))
    # # print(C_pca)
    # plt.subplot(122)
    # plt.scatter(A_pca[:, 0], A_pca[:, 1])
    # plt.show()
