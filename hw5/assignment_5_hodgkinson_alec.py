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
        self.w_in_hidden = np.random.uniform(0.1, 0.2, (N_h, len(self.X[0]))) - 0.2
        self.w_in_hidden[:, 1] = 1
        self.w_hidden_out = np.random.uniform(0.1, 0.2, (N_o, N_h)) - 0.2
        self.w_hidden_out[:, 0] = 1
        self.eta = eta
        # self.w__hidden_out
        # self.output

    def _one_hot(self, y):
        '''
        encodes class into a one hot encoding (binary vector)
        '''
        encoding = np.zeros(self.N_o)
        if self.N_o != 2:
            encoding[y - 1] = 1.0
        else:
            # binary classification is dumb
            if s == -1:
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
        # hidden values and activated hidden values
        self.hidden_vals = np.dot(self.w_in_hidden, x)
        self.h_activated = self._sigmoid(self.hidden_vals)
        # output values and activated output values
        self.output_vals = np.dot(self.w_hidden_out, self.h_activated)
        self.o_activated = self._sigmoid(self.output_vals)
        ind = np.argmax(self.o_activated)
        self.output = np.zeros(np.shape(self.output_vals))
        self.output[ind] = 1
        return self.output

    def _back_prop(self, x, y, y_hat):
        delta_o = y_hat * (1 - y_hat) * (y - y_hat)
        self.w_hidden_out = self.w_hidden_out + self.eta * np.outer(delta_o, self.h_activated)
        delta_h = self.h_activated * (1 - self.h_activated) * np.dot(delta_o, self.w_hidden_out)
        self.w_in_hidden = self.w_in_hidden + self.eta * np.outer(delta_h, self._sigmoid(x))

    def train(self, epochs):
        for epoch in range(epochs):
            # get indices of X/y in random order
            for i in np.random.permutation(np.arange(len(self.X))):
                y_hat = self._feed_forward(self.X[i])
                E = 0.5 * np.linalg.norm(y_hat - self.y[i])
                if E > 0:
                    self._back_prop(self.X[i], self.y[i], y_hat)

    def predict(self, z):
        incorrect = 0
        print(np.shape(self.w_in_hidden))
        print(np.shape(z[:, :-1].T))
        def calculate_value(z):
            hidden_activation = self._sigmoid(np.dot(self.w_in_hidden, z.T))
            output_activation = self._sigmoid(np.dot(self.w_hidden_out, hidden_activation))
            output = np.zeros((self.N_o,1))
            ind = np.argmax(output_activation)
            output[ind] = 1
            return output

        def threshold(z):
            
        hidden_acc = self._sigmoid(np.dot(self.w_in_hidden, z[:, :-1].T))
        print(np.shape(self._sigmoid(np.dot(self.w_in_hidden, z[:, :-1].T))))
        print((self._sigmoid(np.dot(self.w_hidden_out, hidden_acc)).T))
        print((self._sigmoid(np.dot(self.w_hidden_out, hidden_acc)).T))

        # for i in range(len(z)):
        #     # print(z[i])
        #     predicted_val = calculate_value(z[:, :-1])
        #     true_val = self._one_hot(z[:, -1])
        #     if predicted_val != true_val:
        #         incorrect += 1
        return incorrect/len(z)

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
    if fname == "iris.txt":
        data = data.reshape((-1, 5))
    else:
        data = data.reshape((-1, 9))
    return data

if __name__ == "__main__":
    args = parse_arguments()
    train_data = parse_data(args.train)
    train_data = np.hstack((np.ones((np.shape(train_data)[0], 1)), train_data))
    test_data = parse_data(args.test)
    test_data = np.hstack((np.ones((np.shape(test_data)[0], 1)), test_data))

    N_hidden, eta, epochs = int(args.N_h), float(args.eta), int(args.epochs)
    N_classes = len(set(train_data[:, -1]))
    model = Network(train_data, N_hidden, N_classes, eta)
    model.train(epochs)
    error = model.predict(test_data)
    print(error)
