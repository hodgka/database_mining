import numpy as np
import argparse
import re

np.set_printoptions(threshold=np.inf, precision=4, suppress=True)


class Network:
    def __init__(self, data, N_h, N_o, eta):
        self.N_i = len(data[0]) - 1
        self.N_h = N_h
        self.N_o = N_o

        self.X = data[:, :-1]
        self.y = data[:, -1]
        # Initialize weights in [-0.1, 0.1]
        self.w_hidden = np.random.uniform(0.1, 0.2, (N_h, len(self.X[0]) + 1)) - 0.2
        self.w_out = np.random.uniform(0.1, 0.2, (N_o, N_h + 10)) - 0.2
        self.w_hidden[:, 1] = 1
        self.w_out[:, 0] = 1

    def _one_hot(self, y):
        '''
        encodes class into a one hot encoding (binary vector)
        '''
        encoding = np.zeros((N_o))
        if N_o != 2:
            encoding[y - 1] = 1.0
        else:
            # binary classification is dumb
            if y == -1:
                encoding[0] = 1
            else:
                encoding[1] = 1
        self.y_hat = encoding
        return encoding

    def _hone_ot(self, y_hat):
        '''
        converts from one hot encoding to class labels
        '''
        if len(y_hat) == 2:
            if y_hat[0] == 1:
                return -1
            else:
                return 1
        else:
            # finds nonzero entries(there's only 1) and converts it to an int
            return int(np.where(y_hat > 0)) + 1

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _delta(self, o):
        delta_o = o * (1 - o) * (self.y - o)

    def _feed_forward(self, x):
        self.hidden_values = self._sigmoid(np.dot(self.w_h, x))
        self.output = self._sigmoid(np.dot(self.w_o, self.hidden_values))
        ind = np.argmax(self.output)
        self.output_vec = np.zeros(np.shape(output))
        self.output_vec[ind] = 1

    def _back_prop(self, x, y, y_hat):
        delta_out = y_hat * (1 - y_hat) * (self.y - y_hat)
        self.w_out += self.eta * delta_out * y_hat
        delta_hidden = self.o_hidden * (1 - self.o_hidden) * np.dot(delta_out, self.w_out)
        self.w_hidden += self.eta * delta_hidden * self.o_hidden

    def train(self, epochs):
        for epoch in range(epochs):
            # get indices of X/y in random order
            for i in np.random.permutation(np.arange(len(self.X))):
                y_hat = self._feed_forward(self.X[i])
                E = 0.5 * np.linalg.norm(y_hat - self.y[i])
                if E > 0:
                    self._back_prop(self.X[i], self.y[i], y_hat)

    def predict(self, z):
        y_hat = self._feed_forward(z)
        return self._hone_ot(y_hat)






def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument("train")
    args.add_argument("test")
    args.add_argument("N_h")
    args.add_argument("eta")
    args.add_argument("epochs")
    return args.parse_args()

def parse_data(fname):
    with open(fname, 'r') as f:
        d = f.read()
        data = np.array(list(map(float, re.split(r'[,\n]', d)[:-1])))
    if fname == "Concrete_Data_RNorm_Class.txt":
        data = data.reshape((-1, 9))
    else:
        data = data.reshape((-1, 5))
    return data

if __name__ == "__main__":
    args = parse_arguments()
    train_data = parse_data(args.train)
    test_data = parse(args.test)
    N_hidden, eta, epochs = args.N_h, args.eta, args.epochs
