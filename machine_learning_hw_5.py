import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

# np.set_printoptions(precision=6, linewidth=80, threshold=np.inf, suppress=True)
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
    a = (dataset[:, 3] - dataset[:, 1])/(dataset[:, 2] - dataset[:, 0])
    b = dataset[:, 1] + a*dataset[:, 0]
    return np.array([a, b]).T


def g_avg(weights):
    return np.average(weights, axis=0)


def bias(g, x):
    N = np.shape(x)[0]
    x = np.concatenate((x[:, 0], x[:, 2]), axis=0)

    # g[0] is a, g[1] is b
    # multiply g[1] by vector of ones so that you can add them together
    g_bar_values = g[0]*x + g[1]*np.ones(np.shape(x))
    g_minus_f_squared = (g_bar_values - x*x)**2
    return np.sum(g_minus_f_squared)/N


# def bias(dataset, g_weights):
#     N = np.shape(g_weights)[0]
#     d = np.shape(dataset)[0]
#     g_error = np.empty((N, d))
#     for j in range(d):
#         for i in range(N):
#             # print(g_weights[i][0])
#             # print(g_weights[i][0], g_weights[i][1], dataset[j])
#             g_error[i][j] = (g_weights[i][0] * dataset[j] +
#                              g_weights[i][1] - dataset[j]**2)**2
#     g_bias = (1/(d*N)) * np.sum(g_error)
#     print(g_bias)
#     # g_data_range = g(dataset, g_weights)
#     # print(g_data_range)


def var(g, x):
    N = np.shape(x)[0]
    x = np.concatenate((x[:, 0], x[:, 2]), axis=0)
    g_bar_squared = g[0]*x + g[1]*np.ones(np.shape(x))**2
    print(g_bar_squared)
    return

if __name__ == "__main__":
    trials = 50
    dataset = generate_points()
    data_range = np.linspace(-1, 1, 50)

    g_weights = calculate_g_weights(dataset)
    average_hypothesis = g_avg(g_weights)
    fig = plt.figure()
    plt.subplot(111)
    x = np.linspace(-1, 1, 500)
    y1 = average_hypothesis[0]*x + average_hypothesis[1] * np.ones(np.shape(x))
    y2 = x**2
    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'b')
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
        # print(generate_points()[:50, :])
