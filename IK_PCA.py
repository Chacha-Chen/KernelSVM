# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 07:58:36 2018

@author: Dynasting
"""
import Kernel
import numpy as np
#from matplotlib.pyplot import subplots, show
from numpy import linalg as LA
import LSSVM
import Dataset
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
    if kernel_dict['type'] == 'RBF':
        K = Kernel.RBF(m,kernel_dict['gamma'])
        K.calculate(U)
    elif kernel_dict['type'] == 'LINEAR':
        K = Kernel.LINEAR(m)
        K.calculate(U)
    elif kernel_dict['type'] == 'POLY':
        K = Kernel.POLY(m,kernel_dict['c'],kernel_dict['d'])
        K.calculate(U)
    elif kernel_dict['type'] == 'TANH':
        K = Kernel.TANH(m,kernel_dict['c'],kernel_dict['d'])
        K.calculate(U)
    elif kernel_dict['type'] == 'TL1':
        K = Kernel.TL1(m,kernel_dict['rho'])
        K.calculate(U)
    return U, E, V, K

def kPCA_test(Xtest,K,V):
    K.expand(Xtest)
    X_new = np.dot(K.testMat,V)
    return X_new


if __name__ == '__main__':  
    Train = np.loadtxt('dataset///SPECT_train.txt')
    X = Train[:,1:-1]
    Y = Train[:,0]
    NorX = Dataset.Normalization()
    NorX.fit(X)
    NorY = Dataset.Normalization()
    NorY.fit(Y)
    X_N = NorX.fT(X)
    Y_N = NorY.fT(Y)
    Y_N = Y_N * 2 -1
    
    Test = np.loadtxt('dataset///SPECT_test.txt')
    X_t = Test[:,1:-1]
    Y_t = Test[:,0]
    X_N_t = NorX.fT(X_t)  
    Y_N_t = NorY.fT(Y_t)  
    Y_N_t = Y_N_t * 2 -1
    
    kernel_dict = {'type':'RBF','gamma' : 1}
    (U, E, V, K) = pca(X_N, kernel_dict, 3)
    
    (alpha,b,K) = LSSVM.LSSVM_CV(U,Y_N,'LINEAR',[0.001,0.005,0.01,0.05,0.1,0.5,1,4,10,25,100])
    X_new = kPCA_test(X_N_t,K,V)
    
    Y_predict = LSSVM._LSSVMpredict(X_new,K,alpha,b,Y_N)
    acc = LSSVM._compare(Y_N_t,Y_predict)

    print(acc)
    
'''
fig, (ax1, ax2) = subplots(1, 2)
ax1.scatter(data[:50, 0], data[:50, 1], c = 'r')
ax1.scatter(data[50:, 0], data[50:, 1], c = 'b')
ax2.scatter(trans[:50, 0], trans[:50, 1], c = 'r')
ax2.scatter(trans[50:, 0], trans[50:, 1], c = 'b')
show()
'''