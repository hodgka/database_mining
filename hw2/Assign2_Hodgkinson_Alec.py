# Alec Hodgkinson
# Database Mining
# 9/22/2016

####################################
#         Uses Python 3.5          #
####################################

import argparse
import numpy as np
import matplotlib.pyplot as plt
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
    return K


def center_kernel(K):
    n = np.shape(K)[0]
    A = np.eye(n) - np.ones((n, n))/n
    K_centered = np.dot(A, np.dot(K.T, A))
    return K_centered


def normalize_kernel(K):
    w = np.diag(1./np.sqrt(np.diag(K)))
    K_normalized = np.dot(w, np.dot(K.T, w))
    return K_normalized


def dominant_eigenvectors(K, n=0, alpha=None):
    eigen_val, eigen_vec = np.linalg.eigh(K)
    eigen_val = eigen_val[::-1]
    eigen_vec = eigen_vec[:, ::-1]
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
        print("Alpha is None")


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

##########################################################################
##########################################################################

if __name__ == "__main__":
    # open the file with the given filename and put it in an array 1030x9 array

    with open(sys.argv[1], 'r') as f:
        d = f.read()
        data = np.array(list(map(float, re.split(r'[,\n]', d)[:-1])))
        data = data.reshape((-1, 9))

    K = create_kernel(data, "linear")
    K_normalized = normalize_kernel(K)
    C = dominant_eigenvectors(K_normalized, n=2, alpha=0.95)[1]
    # print(C)
    A = reduce_data_dimensionality(K, C)
    print(np.shape(K))
    print((C))
    # for i in range(9):
    #     for i in range():
    #         plt.scatter(data[:, i], data[:, j])
    # plt.scatter(data[:, 8], data[:, 7])
    fig = plt.figure()
    plt.subplot(121)
    plt.scatter(A[0], A[1])
    # plt.show()
    # variance = float(sys.argv[2])
    S = create_covariance(data)
    C_pca = dominant_pca_eigenvectors(S, 2)[1]
    A_pca = np.dot(data, C_pca)
    print(np.shape(A_pca))
    # print(C_pca)
    plt.subplot(122)
    plt.scatter(A_pca[:, 0], A_pca[:, 1])
    plt.show()
