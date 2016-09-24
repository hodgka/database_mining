# Alec Hodgkinson
# Database Mining
# 9/11/2016

####################################
#         Uses Python 2.7          #
####################################

import argparse
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, linewidth=80, suppress=True)

##########################################################################
##########################################################################
# Takes 3 arguments with flags -fname, -func, and -eps(optional). The arguments
# are the filename to parse data from, the function you want to execute, and
# the value of epsilon you want to use.
##########################################################################
##########################################################################
# Possible function arguments include: mean, variance, "covariance inner",
# "covariance outer", correlation, eigenvector, and all
##########################################################################
##########################################################################
# Epsilon value should be given in the format " 10e-10"
##########################################################################
##########################################################################
parser = argparse.ArgumentParser(description="Database Mining Homework 1")
parser.add_argument('-fname', metavar='filename', type=str,
                    default='airfoil_self_noise.txt',
                    help='File to collect data from.')
parser.add_argument('-func', metavar="function", type=str, default='mean',
                    help="Function to call.")
parser.add_argument('-eps', metavar="epsilon", type=float,
                    help="Value for epsilon.")
args = parser.parse_args()


def mean(x):
    '''
    calculates mean of each attribute and returns a numpy array with the means
    x is the raw data
    '''
    n = np.shape(x)[0]
    vals = np.sum(x, axis=0)/n
    return vals

##########################################################################
##########################################################################


def var(x):
    '''
    function to calculate variance of the attributes and returns an array
    x is the raw data
    '''
    n = np.shape(x)[0]
    z = x-mean(x)
    normz2 = z*z
    s = np.sum(normz2, axis=0)
    s /= n
    s = np.sum(s)
    return s

##########################################################################
##########################################################################


def covariance_inner(z):
    '''
    calculates the covariance matrix using inner products and returns the\
     matrix
    z is the centered data
    '''
    # covariance matrix using inner products
    n = np.shape(z)[0]
    sigma = np.dot(z.T, z)
    sigma /= n
    return sigma

##########################################################################
##########################################################################


def covariance_outer(z):
    '''
    calculates the covariance matrix using outer products and returns the\
     matrix
    z is the centered data
    '''
    n = np.shape(z)[0]
    outers = map(lambda zi: np.outer(zi, zi), z)
    return mean(outers)

##########################################################################
##########################################################################


def cosine_dist(z):
    '''
    finds the correlation coefficient for each attribute and returns the \
    correlation matrix

    z is the centered data
    '''
    n = np.shape(z)[0]
    norms = np.empty([n, 1])
    correlations = np.empty([6, 6])
    z /= np.linalg.norm(z, axis=0)
    for i in range(6):
        for j in range(6):
            correlations[i][j] = np.dot(z[:, i].T, z[:, j])
    return correlations

##########################################################################
##########################################################################


def plot_data(x):
    '''function to automaticall generate plots for the data with each pairing \
    of attributes i.e. only 1 graph with attributes A1,A6, etc.

    Saves the plots to a file depending on attribute numbers
    '''
    fig = plt.figure()
    n = np.shape(x)[0]
    for i in xrange(6):
        for j in xrange(i+1, 6):
            plt.scatter(x[:, i], x[:, j])
            plt.savefig("Assign1_Hodgkinson_Alec{0}{1}.png".format(i+1, j+1))
            plt.clf()
    plt.show()

##########################################################################
##########################################################################


def dominant_eigenvector(sigma, epsilon=10**(-10)):
    '''
    find the dominant value and vector using iteration method
    returns a tuple with the eigenvalue and eigenvector respectively
    sigma is the covariance matrix
    epsilon is the accuracy. Defaults to 10^-10
    '''
    # current and previous x vectors
    x = np.ones((np.shape(sigma)[0], 1))
    prev = np.zeros(np.shape(x))
    # make sure it doesn't loop forever
    iteration = 0
    m = 0
    while np.linalg.norm(x-prev) > epsilon and iteration < 100:
        prev = x
        x = np.dot(sigma, x)
        m = np.amax(x)
        x /= m
        iteration += 1
    x /= np.linalg.norm(x)
    largest_eigenvalue = (x/prev)/m
    return (m, x)

##########################################################################
##########################################################################


def project_data(x, u):
    '''
    projects the data onto a vector, u. returns that projected dataset
    x is the data
    u is the vector to be projected onto
    returns a matrix with the same shape as x, but projected onto u
    '''

    scalar_proj = np.dot(x, u)/float(np.dot(u.T, u))

    scalar_proj = scalar_proj.flatten()
    ###############################################
    # if you want the actual projected data values#
    ###############################################

    # print np.shape(scalar_proj)
    # vector_proj = np.empty(np.shape(x))
    # for i in range(len(scalar_proj)):
    #     vector_proj[i] = np.multiply(scalar_proj[i],u.T)
    # return vector_proj

    return scalar_proj

##########################################################################
##########################################################################

if __name__ == "__main__":
    # open the file with the given filename and put it in an array nx6 array
    with open(args.fname, 'r') as f:
        d = f.read()
        d = d.strip().split()
        data = np.array(list(map(float, d)))
        data = data.reshape((-1, 6))

    z = data-mean(data)
    if args.func == "all":
        mu = mean(data)
        var = var(data)
        inner_cov = covariance_inner(z)
        outer_cov = covariance_outer(z)
        cor = cosine_dist(z)

        if args.eps is not None:
            eigen = dominant_eigenvector(inner_cov, args.eps)
        else:
            eigen = dominant_eigenvector(inner_cov)
        eigenvalue = eigen[0]
        eigenvector = np.array(eigen[1].T)[0]
        x_projected = project_data(data, eigenvector)

        print "Mu: ", mu
        print "Variance: ", var
        print
        print "Covariance Matrix(Inner): "
        print inner_cov
        print
        print "Covariance Matrix(Outer): "
        print outer_cov
        print
        print "Correlation Matrix: "
        print cor
        print
        print "Largest EigenValue: ", eigenvalue
        print "Largest Eigenvector: "
        print eigenvector
        print
        np.set_printoptions(threshold=np.inf)
        print "Projected Data: "
        print x_projected
        with open("Assign1-Hodgkinson-Alec.txt", 'w') as f:
            f.write("PART 1 \n\n\n\nMean: {0} \n\nVariance: {1}\n\n\
                    Covariance(Inner Product):\n {2}\n\n\
                    Covariance(Outer Product):\n {3}\n\n\
                    Correlation: \n{4}\n\n\n\nPART 2\n\n\
                    Eigenvalue: {5}\n\nEigenvector: {6}\n\n\
                    Projected Data: \n{7}\n".format(mu, var,
                                                    inner_cov, outer_cov, cor,
                                                    eigenvalue, eigenvector,
                                                    x_projected))
        plot_data(data)

    if args.func == "mean":
        print "Mean: ", mean(data)
        print "Numpy Mean: ", np.mean(data, axis=0)

    if args.func == "variance":
        print "Total Variance: ", var(data)
        print "Numpy Total Variance: ", np.sum(np.var(data, axis=0,))

    if args.func == "covariance inner":
        print "Covariance Matrix with Inner Products: ", covariance_inner(z)
        print "Numpy Covariance Matrix: ", np.cov(data.T, bias=True)

    if args.func == "covariance outer":
        print "Covariance Matrix with Outer Products: ", covariance_outer(z)
        print "Numpy Covariance Matrix: ", np.cov(data.T, bias=True)

    if args.func == "correlation":
        print cosine_dist(z)

    if args.func == "eigenvector":
        # create covariance matrix
        sigma = covariance_inner(z)
        if args.eps is not None:
            eigen = dominant_eigenvector(sigma, args.eps)
        else:
            eigen = dominant_eigenvector(sigma)
        eigenvalue = eigen[0]
        eigenvector = np.array(eigen[1].T)[0]

        numpyeigvals, numpyeigvecs = np.linalg.eig(sigma)
        m = numpyeigvals.argmax()
        x_projected = project_data(data, eigenvector)
        print "Largest EigenValue and EigenVector: ", eigenvalue
        print eigenvector
        print
        print "Numpy Eigenvalues and EigenVectors: ", numpyeigvals[m]
        print numpyeigvecs[:, AssertionErrorm]
        np.set_printoptions(threshold=np.inf)
        print
        print "Projected Data: "
        print x_projected
