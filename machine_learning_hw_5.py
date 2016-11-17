import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

# np.set_printoptions(precision=6, linewidth=80, threshold=np.inf,
#                     suppress=True)
np.set_printoptions(precision=6, linewidth=80, suppress=True)


###########################
#     Part Something      #
###########################
# fig, ax = plt.subplots(1)
# # x = np.arange(-2, 2, 100)
# # y = lambda x: -x
# # y = np.vectorize(y)
# # y = y(x)
# # plt.plot([-2, 2], [2, -2])
# # plt.plot([1, 1.5, -0.5], [1, -0.5, 1.5], 'or', label="positive (+1)")
# # plt.plot([-1, -1.5, 0.33], [-1, -.7, -1.33], 'xb', label="negative (-1)")
# # plt.legend()
# # plt.show()
# plt.plot([-2, -1], [2, 2], 'b')
# plt.plot([-1, -1], [2, 1], 'b')
# plt.plot([-1, 0], [1, 1], 'b')
# plt.plot([0, 0], [1, 0], 'b')
# plt.plot([0, 1], [0, 0], 'b')
# plt.plot([1, 1], [0, -1], 'b')
# plt.plot([1, 2], [-1, -1], 'b')
# plt.plot([2, 2], [-1, -2], 'b')
# plt.plot([1, 0.1, 1.1], [1, 0.1, -.9], 'or', label="positive (+1)")
# plt.plot([-1, -0.1, -1.1], [-1, 0.1, 1.1], 'xb', label="negative (-1)")
# plt.legend()
# plt.show()

###########################
#        Part 2.24        #
###########################


def generate_points(N=100000):
    points = random.uniform(-1, 1, 2*N)
    points = points.reshape((-1, 2))
    all_values = np.empty((len(points), 4))
    for i in range(len(points)):
        all_values[i] = np.array([points[i][0],
                                  points[i][0]**2,
                                  points[i][1],
                                  points[i][1]**2])
    return all_values


def calculate_g_weights(dataset):
    '''
    get weights for g = ax +b, between two datapoints for a dataset
    '''
    a = (dataset[:, 3] - dataset[:, 1])/(dataset[:, 2] - dataset[:, 0])
    b = dataset[:, 1] + a*dataset[:, 0]
    return np.array([a, b]).T


def g_avg(weights):
    '''
    calculate average set of weights given all weights for g
    '''
    return np.average(weights, axis=0)


def bias(g, x):
    '''
    calculate the total bias for Eout
    '''

    x = np.concatenate((x[:, 0], x[:, 2]), axis=0)
    N = np.shape(x)[0]
    # g[0] is a, g[1] is b
    # multiply g[1] by vector of ones so that you can add them together
    g_bar_values = g[0]*x + g[1]*np.ones(np.shape(x))
    print("g_bar_values", g, g_bar_values)
    g_minus_f_squared = (g_bar_values - x*x)**2
    return np.sum(g_minus_f_squared)/N


def var(g, weights, x):
    '''
    calculate variance for Eout
    '''
    # x = np.concatenate((x[:, 0], x[:, 2]), axis=0)
    N = np.shape(x)[0]
    # print(x)

    # g_bar_squared = np.average(g[0]*x + g[1]*np.ones(np.shape(x)))**2
    gk = 0

    weights = weights - g
    print(np.average(weights[:, 0]*x + weights[:, 1]*np.ones(np.shape(x))))
    # for i in weights:
    #     weights[i][0]**2 x**2 + 2*weights[i][0]*weights[i][1] * x +
    #  weights[i][1]**2 * np.ones(np.shape(x))
    # # print(g_squared - g_bar_squared)
    # print("g_bar", g_bar_squared)
    # print("squared", g_squared)
    # print(np.var())
    return

if __name__ == "__main__":
    trials = 10
    points = 100000
    # create a dataset of 2N points in [-1,1]
    # dataset = generate_points(points)
    # calculate the weights of each g on the dataset
    # g_weights = calculate_g_weights(dataset)
    # get the average weights for all the g on the dataset
    # average_hypothesis = g_avg(g_weights)
    data_points = np.empty((points*trials, 4))
    weights = np.empty((points*trials, 2))
    print(np.shape(weights))
    for i in range(trials):
        dataset = generate_points(points)
        data_points[i*points: (i+1)*points, :] = dataset
        weights[i*points: (i+1)*points, :] = calculate_g_weights(dataset)
    g_bar = g_avg(weights)
    bias_value = bias(g_bar, data_points)
    var_value = var(g_bar, weights, data_points)
    print(bias_value)
    # print(var_value)
    fig = plt.figure()
    plt.subplot(111)
    x = np.linspace(-1, 1, 500)
    y1 = g_bar[0]*x + g_bar[1] * np.ones(np.shape(x))
    y2 = x**2
    plt.plot(x, y1, 'r', label="g(x)")
    plt.plot(x, y2, 'b', label="f(x)")
    plt.legend()
    plt.show()
    # sample_bias = bias(average_hypothesis, dataset)
    # sample_var = var(average_hypothesis, dataset)
    # bias(data_range, g_weights)
    # print(calculate_g_weights(dataset))
    # print(dataset[0], dataset[1])
    # for i in range(trials):
    #     data = generate_points()
    #     g_weights = calculate_g_weights(data)
    #     bias(data_range, g_weights)
    #     # print(generate_points()[:50, :])
