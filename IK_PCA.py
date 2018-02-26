# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 07:58:36 2018

@author: Dynasting
"""
import Kernel
import numpy as np
from matplotlib.pyplot import subplots, show
from numpy import linalg as LA

'''
w, v = LA.eig(np.diag((1, 2, 3)))

'''
#from matplotlib.pyplot import subplots, show


def pca(X, kernel_dict, pc_count = None):
    """
    Principal component analysis using eigenvalues
    note: this mean-centers and auto-scales the data (in-place)
    """
    m = X.shape[0]
    # Kernel
    if kernel_dict['type'] == 'RBF':
        K = Kernel.RBF(m,kernel_dict['gamma'])
        K.calculate(X)
    elif kernel_dict['type'] == 'LINEAR':
        K = Kernel.LINEAR(m)
        K.calculate(X)
    elif kernel_dict['type'] == 'POLY':
        K = Kernel.POLY(m,kernel_dict['c'],kernel_dict['d'])
        K.calculate(X)
    elif kernel_dict['type'] == 'TANH':
        K = Kernel.TANH(m,kernel_dict['c'],kernel_dict['d'])
        K.calculate(X)
    elif kernel_dict['type'] == 'TL1':
        K = Kernel.TL1(m,kernel_dict['rho'])
        K.calculate(X)
    
    
    # Construct Kernel Matrix
    print(K.kernelMat)
    
    # Centrelize
    C = K.kernelMat - K.kernelMat.mean(0) - np.matrix(K.kernelMat.mean(1)).T + K.kernelMat.mean()
    
    
    E, V = LA.eigh(C)
    key = np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = np.dot(C, V)
    return U, E, V, K

def kPCA_test(Xtest,K,V):
    K.expand(Xtest)
    X_new = np.dot(K.testMat,V)
    return X_new


if __name__ == '__main__':
    X = np.loadtxt('dataset//monks_2_train.txt')
    kernel_dict = {'type':'RBF','gamma' : 1}
    (U, E, V, K) = pca(X, kernel_dict, 3)
    
    X2 = np.loadtxt('dataset//monks_2_test.txt')
    X_new = kPCA_test(X2,K,V)
    
'''
fig, (ax1, ax2) = subplots(1, 2)
ax1.scatter(data[:50, 0], data[:50, 1], c = 'r')
ax1.scatter(data[50:, 0], data[50:, 1], c = 'b')
ax2.scatter(trans[:50, 0], trans[:50, 1], c = 'r')
ax2.scatter(trans[50:, 0], trans[50:, 1], c = 'b')
show()
'''