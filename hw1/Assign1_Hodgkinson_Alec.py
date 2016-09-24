# Alec Hodgkinson
# Database Mining
# 9/11/2016

####################################
#         Uses Python 2.7          #
####################################

import sys
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, linewidth=80, suppress=True)

##########################################################################
##########################################################################
# Takes 3 arguments with flags -fname, -func, and -eps(optional). The arguments are
# the filename to parse data from, the function you want to execute, and
# the value of epsilon you want to use.
##########################################################################
##########################################################################
# Valid kernel functions: linear, gaussian
##########################################################################
##########################################################################
# Epsilon value should be given in the format " 10e-10"
##########################################################################
##########################################################################
# parser = argparse.ArgumentParser(description="Database Mining Homework 2")
# parser.add_argument('-fname', metavar='filename', type=str,
#                     default='Assign2.txt',
#                     help='File to get data from.')
# parser.add_argument('-kern', metavar="kernel", type=str, default='linear',
#                     help="Type of kernel to use for PCA. Linear or Gaussian")
# # parser.add_argument('-eps', metavar="epsilon", type=float,
# #                     help="Value for epsilon.")
# args = parser.parse_args()



##########################################################################
##########################################################################

##########################################################################
##########################################################################

def plot_data(x):
    '''function to automaticall generate plots for the data with each pairing of attributes i.e. only 1 graph with attributes A1,A6, etc.

    Saves the plots to a file depending on attribute numbers
    '''
    fig = plt.figure()
    n = np.shape(x)[0]
    for i in xrange(6):
        for j in xrange(i+1,6):
            plt.scatter(x[:,i],x[:,j])
            plt.savefig("Assign1_Hodgkinson_Alec{0}{1}.png".format(i+1,j+1))
            plt.clf()
    plt.show()

##########################################################################
##########################################################################

if __name__ == "__main__":
    # open the file with the given filename and put it in an array nx6 array
    with open(sys.argv[1],'r') as f:
        d = f.read()
        d = d.strip().split()
        data = np.array(list(map(float,d)))
        data = data.reshape((-1,6))
    print(data)
    # z = data-mean(data)
    # if args.func == "all":
    #     mu = mean(data)
    #     var = var(data)
    #     inner_cov = covariance_inner(z)
    #     outer_cov = covariance_outer(z)
    #     cor = cosine_dist(z)
    #
    #     if args.eps is not None:
    #         eigen = dominant_eigenvector(inner_cov, args.eps)
    #     else:
    #         eigen = dominant_eigenvector(inner_cov)
    #     eigenvalue = eigen[0]
    #     eigenvector = np.array(eigen[1].T)[0]
    #     x_projected = project_data(data,eigenvector)
    #
    #
    #     print "Mu: ",mu
    #     print "Variance: ",var
    #     print
    #     print "Covariance Matrix(Inner): "
    #     print inner_cov
    #     print
    #     print "Covariance Matrix(Outer): "
    #     print outer_cov
    #     print
    #     print "Correlation Matrix: "
    #     print cor
    #     print
    #     print "Largest EigenValue: ", eigenvalue
    #     print "Largest Eigenvector: "
    #     print eigenvector
    #     print
    #     np.set_printoptions(threshold=np.inf)
    #     print "Projected Data: "
    #     print x_projected
    #     with open("Assign1-Hodgkinson-Alec.txt",'w') as f:
    #         f.write("PART 1 \n\n\n\nMean: {0} \n\nVariance: {1}\n\nCovariance(Inner Product):\n {2}\n\nCovariance(Outer Product):\n {3}\n\nCorrelation: \n{4}\n\n\n\nPART 2\n\nEigenvalue: {5}\n\nEigenvector: {6}\n\nProjected Data: \n{7}\n".format(mu, var,
    #         inner_cov, outer_cov, cor, eigenvalue, eigenvector,x_projected))
    #     plot_data(data)
    #
    # if args.func == "mean":
    #     print "Mean: ", mean(data)
    #     print "Numpy Mean: ", np.mean(data, axis=0)
    #
    # if args.func == "variance":
    #     print "Total Variance: ",var(data)
    #     print "Numpy Total Variance: ",np.sum(np.var(data, axis=0,))
    #
    # if args.func == "covariance inner":
    #     print "Covariance Matrix with Inner Products: ",covariance_inner(z)
    #     print "Numpy Covariance Matrix: ", np.cov(data.T, bias=True)
    #
    # if args.func == "covariance outer":
    #     print "Covariance Matrix with Outer Products: ",covariance_outer(z)
    #     print "Numpy Covariance Matrix: ", np.cov(data.T, bias=True)
    #
    # if args.func == "correlation":
    #     print cosine_dist(z)
    #
    # if args.func == "eigenvector":
    #     # create covariance matrix
    #     sigma = covariance_inner(z)
    #     if args.eps is not None:
    #         eigen = dominant_eigenvector(sigma,args.eps)
    #     else:
    #         eigen = dominant_eigenvector(sigma)
    #     eigenvalue = eigen[0]
    #     eigenvector = np.array(eigen[1].T)[0]
    #
    #     numpyeigvals,numpyeigvecs = np.linalg.eig(sigma)
    #     m = numpyeigvals.argmax()
    #     x_projected = project_data(data,eigenvector)
    #     print "Largest EigenValue and EigenVector: ", eigenvalue
    #     print eigenvector
    #     print
    #     print "Numpy Eigenvalues and EigenVectors: ", numpyeigvals[m]
    #     print numpyeigvecs[:,m]
    #     np.set_printoptions(threshold=np.inf)
    #     print
    #     print "Projected Data: "
    #     print x_projected
