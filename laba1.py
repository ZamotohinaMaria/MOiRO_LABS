import numpy as np
import math
import random
import matplotlib.pyplot as plt

def GenerateRandomVectors(meanVector, covarianceMatrix, N, savedFileName):
    # Transformation matrix for generation
    A = np.zeros((2,2))
    A[0,0] = math.sqrt(covarianceMatrix[0, 0])
    A[1,0] = covarianceMatrix[1, 0] / math.sqrt(covarianceMatrix[0, 0])
    A[1,1] = math.sqrt(covarianceMatrix[1, 1] - covarianceMatrix[0, 1] * covarianceMatrix[0, 1] / covarianceMatrix[0, 0])
    
    # Generate random normal vectors
    vectorsNorm01 = np.random.randn(2, 1)  # (0,1) normal vectors
    
    # Generated vectors with given mean and covariance
    x = np.matmul(A, vectorsNorm01) + np.repeat(meanVector, N, axis=1)
    
    # Save generated vectors to file
    np.save(savedFileName, x)
    
    # Optional plot to check
    # plt.plot(x[0,:], x[1,:], color='green', marker='x', linestyle='none')
    # plt.show()

# Parameters
N = 100
M = np.array([[5],[4]])  # mean vector
B = np.array([[5, 2], [2, 1]])  # covariance matrix
fileName2Save = 'arrayX.npy'

# Generate and save vectors
GenerateRandomVectors(M, B, N, fileName2Save)

# Load and plot saved data
z = np.load(fileName2Save)
plt.plot(z[0,:], z[1,:], color='red', marker='.', linestyle='none')  # plot saved and loaded data
plt.show()
