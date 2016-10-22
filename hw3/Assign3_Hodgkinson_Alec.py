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
        K = np.linalg.matrix_power(np.dot(x, x.T), 2)

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

    alpha = np.zeros(np.shape(x)[0])
    alpha_prev = np.ones(np.shape(alpha))
    tol = 10**-5
    tryall = True
    while np.linalg.norm(alpha-alpha_prev) > eps:
        it += 1
        print("Running iteration: {0}".format(it))

        alpha_prev = np.copy(alpha)
        for j in range(n):
            if tryall is False and (alpha[j]-tol < 0 or alpha[j]+tol > C):
                continue
            for i in np.random.random_integers(0, n-1, n):
                if i == j:
                    continue
                if tryall is False and (alpha[i]-tol < 0 or alpha[i]+tol > C):
                    continue
                kij = K[i, i] + K[j, j] - 2*K[i, j]
                if kij == 0:
                    continue
                alpha_i = alpha[i]
                alpha_j = alpha[j]
                L, H = compute_L_H(y[i], y[j], C, alpha_i, alpha_j)
                if L == H:
                    continue
                alpha[j] = alpha_j + y[j]*compute_E_ij(alpha, y, K, i, j)/kij
                if alpha[j] < L:
                    alpha[j] = L
                elif alpha[j] > H:
                    alpha[j] = H
                # print(alpha_i, y[i], y[j], alpha_j, alpha[j])
                alpha[i] = alpha_i + y[i]*y[j]*(alpha_j - alpha[j])
        if tryall:
            tryall = False
    alpha_filtered = alpha[(tol < alpha) & (alpha < C-tol)]
    return (alpha, alpha_filtered)


def compute_L_H(yi, yj, C, ai, aj):
    if yi != yj:
        L = max(0, aj-ai)
        H = min(C, C - ai + aj)
    else:
        L = max(0, aj + ai - C)
        H = min(C, ai + aj)
    return (L, H)


def compute_E_ij(alpha, y, K, i, j):
    ei1 = 0
    ej1 = 0
    alpha[alpha < 0] = 0
    for p in range(len(y)):
        ei1 += alpha[p]*y[p]*K[p, i]
        ej1 += alpha[p]*y[p]*K[p, j]
    ei1 -= y[i]
    ej1 -= y[j]

    # ei2 = 0
    # ei2 = np.sum(alpha*y*K[:, i])
    # ei2 -= y[i]
    # ej2 = 0
    # ej2 = np.sum(alpha*y*K[:, j])
    # ej2 -= y[j]
    # ei = 0
    # ej = 0
    # for l in range(len(y)):
    #     ei += alpha[l]*y[l]*K[l, i]
    #     ei += alpha[l]*y[l]*K[l, j]
    # print(ei - ej)
    return ei1-ej1


def get_support_vectors(data, alpha):
    alpha[alpha < 0] = 0
    indices = np.array([i for i in range(len(alpha)) if alpha[i] != 0])
    svs = data[indices]

    return svs


def compute_bias(data, alpha, K, C, tol):
    y = data[:, -1]
    y[alpha <= 0] = 0
    alpha[alpha <= 0] = 0
    n = np.count_nonzero(alpha)
    b = np.empty(np.shape(y))
    for i in range(len(y)):
        b[i] = y[i] - np.sum(alpha*y*K[:, i])
    b_avg = np.average(b)
    return b_avg


def compute_accuracy(data, K, alpha, b, C, tol):
    x = data[:, :-1]
    y = data[:, -1]
    alpha[alpha < 0] = 0
    alpha[alpha > C-tol] = 0
    y[alpha == 0] = 0
    error = 0
    yhat = np.empty(np.shape(y))
    for j in range(len(y)):
        temp = 0
        for i in range(len(alpha)):
            temp += alpha[i]*y[i] * K[i, j]
            # print(alpha[i], y[i], K[i, 0])
        temp += b
        yhat[j] = np.sign(temp)

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
    alpha, alpha_f = SMO(K, data, C, eps)
    svs = get_support_vectors(data, alpha_f)
    b = compute_bias(data, alpha, K, C, 10**-5)
    acc = compute_accuracy(data, K, alpha, b, C, 10**-5)
    if args.w is not None:
        with open("Assign3_Hodgkinson_Alec.txt", 'w') as f:
            f.write("Kernel type: {0}\n Support Vectors: {1}\n Bias: {2}\n\
                    Accuracy {3}".format(args.kernel_type, svs, b, acc))
    print("Kernel Type: {0}")
    print("Support Vector Lagrange Multipliers: ", alpha_f)
    print("Support vectors: \n", svs)
    print("Bis: ", b)
    print("Accuracy: ", acc)
