###################
# Alec Hodgkinson #
# Homework 4      #
# Database Mining #
# Using Python 3.5#
###################

import re
import numpy as np
import pprint
import argparse

np.set_printoptions(suppress=True,
                    precision=4,
                    threshold=np.inf,
                    linewidth=120)


class LogisticRegression:
    def __init__(self, data, eta=0.01, eps=0.01, iterations=2000):
        self.n = np.shape(data)[0]
        self.d = np.shape(data)[1] - 1
        self.x = data[:, :-1]
        self.y = data[:, -1]
        self.w = np.random.random((self.d, 1))
        self.eta = eta
        self.eps = eps
        self.iterations = iterations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sgd(self):
        w = self.w
        w_prev = np.zeros(np.shape(w))
        iteration = 0
        while np.linalg.norm(w - w_prev) > self.eps:
            w_prev = w
            # have to do this because shuffle returns None
            # indices = list(range(self.n))
            # np.random.shuffle(indices)
            # for k in indices:
            for k in range(self.n):
                temp = self.eta * self.sigmoid(-self.y[k] * np.dot(self.x[k], w)
                                               ) * self.y[k] * self.x[k]
                temp = temp.reshape(np.shape(w))
                w += temp
        self.w = w
        return w

    def predict(self, z):
        # print(np.shape(self.w))
        if self.sigmoid(np.dot(z, self.w)) < 0.5:
            return -1
        return 1

    def accuracy(self, test_set):
        num_correct = 0
        self.correct = []
        self.incorrect = []
        for data_point in test_set:
            if self.predict(data_point[:-1]) == data_point[-1]:
                num_correct += 1
                self.correct.append(data_point)
            else:
                self.incorrect.append(data_point)
        return num_correct / len(test_set)


def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument("train_file")
    args.add_argument("test_file")
    args.add_argument("eps")
    args.add_argument("eta")
    return args.parse_args()


def parse_data(fname):
    with open(fname) as f:
        d = f.read()
        # splits inputs by comma and newline delimiters. empty string at end.
        data = np.array(list(map(float, re.split(r'[,\n]', d)[:-1])))
    data = data.reshape((-1, 9))
    data = np.hstack((np.ones((len(data), 1)), data))
    return data

if __name__ == "__main__":
    arguments = arg_parser()
    training_data = parse_data(arguments.train_file)
    testing_data = parse_data(arguments.test_file)
    model = LogisticRegression(training_data,
                               float(arguments.eta),
                               float(arguments.eps))
    weights = model.sgd()
    accuracy = model.accuracy(testing_data)
    print("Weights: \n", weights)
    print()
    print("Accuracy: \n", accuracy)
    print()
    pp = pprint.PrettyPrinter(indent=4)
    print("Correctly classified: ")
    pp.pprint(model.correct)
    print()
    print("Incorrectly classified: ")
    pp.pprint(model.incorrect)
