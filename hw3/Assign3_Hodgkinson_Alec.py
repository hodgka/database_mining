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
    parser.add_argument('-w', default=False, nargs="?")

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


def compute_K(data, t=None):
    x = data[:, :-1]
    y = data[:, -1]
    if t is None or t.lower() == "linear":
        K = np.dot(x, x.T)
    elif t.lower() == 'quadratic':
        # K = np.linalg.matrix_power(np.dot(x, x.T), 2)
        xxt = np.dot(x, x.T)
        K = np.dot(xxt, xxt.T)

    else:
        K = gaussian_kernel(x, spread)
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
            return K
        i += 1


def SMO(K, data, C, eps):
    n = len(data)
    it = 0
    x = data[:, :-1]
    y = data[:, -1]

    a = np.zeros(np.shape(x)[0])
    a_prev = np.ones(np.shape(a))
    tol = 10**-5
    tryall = True
    while np.linalg.norm(a-a_prev) > eps:
        it += 1
        print("Running iteration: {0}".format(it))

        a_prev = np.copy(a)
        for j in range(n):
            if tryall is False and (a[j]-tol < 0 or a[j]+tol > C):
                continue
            for i in np.random.random_integers(0, n-1, n):
                if i == j:
                    continue
                if tryall is False and (a[i]-tol < 0 or a[i]+tol > C):
                    continue
                kij = K[i, i] + K[j, j] - 2*K[i, j]
                if kij == 0:
                    continue
                a_i = a[i]
                a_j = a[j]
                L, H = compute_L_H(y[i], y[j], C, a_i, a_j)
                if L == H:
                    continue
                a[j] = a_j + y[j]*compute_E_ij(a, y, K, i, j)/kij
                if a[j] < L:
                    a[j] = L
                elif a[j] > H:
                    a[j] = H
                # print(a_i, y[i], y[j], a_j, a[j])
                a[i] = a_i + y[i]*y[j]*(a_j - a[j])
        if tryall:
            tryall = False
    a_filtered = a[(tol < a) & (a < C-tol)]
    return (a, a_filtered)


def compute_L_H(yi, yj, C, ai, aj):
    if yi != yj:
        L = max(0, aj-ai)
        H = min(C, C - ai + aj)
    else:
        L = max(0, aj + ai - C)
        H = min(C, ai + aj)
    return (L, H)


def compute_E_ij(a, y, K, i, j):
    ei = np.sum(a*y*K[:, i]) - y[i]
    ej = np.sum(a*y*K[:, j]) - y[j]
    return ei1-ej1


def get_support_vectors(data, a):
    a[a < 0] = 0
    indices = np.array([i for i in range(len(a)) if a[i] != 0])
    svs = data[indices]

    return svs


def compute_bias(y, a, K, C, tol):
    ytemp = np.copy(y)
    ytemp[a <= 0] = 0
    b = ytemp - np.sum(alpha*ytemp*K[:, :], axis=0)
    return np.mean(b)


def compute_accuracy(y, K, a, b, C, tol):
    y_temp = np.copy(y)
    y_temp[a <= 0] = 0
    y_temp[a > C-tol] = 0
    yhat = np.sign(np.sum(a*y_temp*K[:, :], axis=0) + b)
    # for j in range(len(y)):
    #     temp = 0
    #     for i in range(len(a)):
    #         temp += a[i]*y[i] * K[i, j]
    #         # print(a[i], y[i], K[i, 0])
    #     temp += b
    #     yhat[j] = np.sign(temp)

    accuracy = np.sum([1 for k in range(len(y))
                       if y[k] != yhat[k]])/len(y)
    return accuracy


if __name__ == "__main__":
    args = parse_arguments()
    data = parse_data(args.filename)
    C = float(args.C)
    eps = float(args.eps)
    K = compute_K(data, args.kernel_type)
    spread = float(args.spread)
    x = data[:, :-1]
    y = data[:, -1]
    # print(SMO(K, data, C, eps))
    a, a_f = SMO(K, data, C, eps)
    svs = get_support_vectors(data, a_f)
    b = compute_bias(data, a, K, C, 10**-5)
    acc = compute_accuracy(data, K, a, b, C, 10**-5)
    if args.w is not None:
        with open("Assign3_Hodgkinson_Alec.txt", 'w') as f:
            f.write("Kernel type: {0}\n Support Vectors: {1}\n Bias: {2}\n\
                    Accuracy {3}".format(args.kernel_type, svs, b, acc))
    print("Kernel Type: {0}")
    print("Support Vector Lagrange Multipliers: ", a_f)
    print("Support vectors: \n", svs)
    print("Bis: ", b)
    print("Accuracy: ", acc)
