import numpy as np
import matplotlib.pyplot as plt
import argparse
import re


np.set_printoptions(threshold=np.inf, precision=4, suppress=True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('C')
    parser.add_argument('eps')
    parser.add_argument('kernel_type')
    parser.add_argument('spread', default=1, nargs="?")
    return parser.parse_args()


def parse_data(fname):
    with open(fname, 'r') as f:
        d = f.read()
        data = np.array(list(map(float, re.split(r'[,\n]', d)[:-1])))
    if fname == "Concrete_Data_RNorm_Class.txt":
        data = data.reshape((-1, 9))
    else:
        data = data.reshape((-1, 5))
    return data


def SMO(K, data, C, eps):
    n = len(data)
    it = 0
    iterations = 2000
    x = data[:, :-1]
    y = data[:, -1]
    alpha = np.zeros(np.shape(x)[0])
    alpha_prev = np.ones(np.shape(alpha))
    tol = 10**-5
    tryall = True
    while np.linalg.norm(alpha-alpha_prev) > eps and it < iterations:
        it += 1
        print("Running iteration: {0}".format(it))

        alpha_prev = alpha
        for j in range(n):
            a_tol_zero = (alpha-tol) < 0
            a_tol_c = (alpha + tol) > C
            if not tryall and ((alpha[j]-tol) < 0 or (alpha[j] + tol) > C):
                continue
            for i in sorted(range(n), key=lambda q: np.random.random()):
                if i == j:
                    continue
                if not tryall and ((alpha[i]-tol) < 0 or (alpha[i] + tol) > C):
                    continue
                kij = K[i, i] + K[j, j] - 2*K[i, j]
                if kij == 0:
                    continue
                alpha_i = alpha[i]
                alpha_j = alpha[j]
                L, H = compute_L_H(y[i], y[j], C, alpha_i, alpha_j)
                alpha[j] = alpha_j + y[j]*compute_E_ij(alpha, y, K, i, j)/kij
                if alpha[j] < L:
                    alpha[j] = L
                elif alpha[j] > H:
                    alpha[j] = H
                alpha[i] = alpha_i + y[i]*y[j]*(alpha_j - alpha[j])
        if tryall:
            tryall = False
    # alpha = alpha[(tol < alpha) & (alpha < C-tol)]
    return alpha


def compute_L_H(yi, yj, C, ai, aj):
    if yi != yj:
        L = max(0, aj-ai)
        H = min(C, C - ai + aj)
        return (L, H)
    else:
        L = max(0, aj + ai - C)
        H = min(C, ai + aj)
        return (L, H)


def compute_E_ij(alpha, y, K, i, j):
    ei = np.sum(alpha*y*K[:, i])
    ei -= y[i]

    ej = np.sum(alpha*y*K[:, j])
    ej -= y[j]
    # ei = 0
    # ej = 0
    # for l in range(len(y)):
    #     ei += alpha[l]*y[l]*K[l, i]
    #     ei += alpha[l]*y[l]*K[l, j]
    return ei-ej


def compute_bias(x, y, alpha, K):
    alpha[alpha <= 0] = 0
    # inds = np.array([i if alpha[i] > 0 for i in range(len(alpha))])
    b = y - np.sum(alpha*y*k[:, :], axis=0)
    b = np.sum(b)/n
    return b


def compute_K(data, t=None):
    x = data[:, :-1]
    y = data[:, -1]
    if t is None or t.lower() == "linear":
        K = np.dot(data, data.T)
    elif t.lower() == 'quadratic':
        K = np.linalg.matrix_power(np.dot(data, data.T), 2)
    else:
        K = gaussian_kernel(data, spread)
    return K


def gaussian_kernel(data, spread):
    n, d = np.shape(data)
    K = np.zeros(shape=(n, n))

    def gaussian(x, y, spread):
        return np.exp((-(np.linalg.norm(x - y)**2)) / (2 * spread**2))

    X = np.asarray(data)
    i = 0
    for x in X:
        j = 0
        for y in X:
            K[i, j] = gaussian(x.T, y.T, spread)
            j += 1
        i += 1
    return K


def get_support_vectors(data, alpha):
    indices = np.array([i for i in range(len(alpha)) if alpha[i] != 0])
    print(indices)
    support_vectors = data[indices]
    plt.scatter(data[:, 0], data[:, 4], c='b')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 4], c='r')
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    data = parse_data(args.filename)
    C = float(args.C)
    eps = float(args.eps)
    K = compute_K(data, args.kernel_type)
    spread = float(args.spread)
    # print(SMO(K, data, C, eps))
    alpha = SMO(K, data, C, eps)
    get_support_vectors(data, alpha)
