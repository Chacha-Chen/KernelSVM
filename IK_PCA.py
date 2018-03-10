# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 07:58:36 2018

@author: Dynasting
"""
import Kernel
import numpy as np
import LSSVM
import Dataset
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
    #print(K.kernelMat)
    
    # Centrelize
    C = K.kernelMat - K.kernelMat.mean(0) - np.matrix(K.kernelMat.mean(1)).T + K.kernelMat.mean()
    
    
    E, V = LA.eigh(C)
    key = np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    #U = np.dot(C, V)
    U = np.dot(K.kernelMat, V)
    #E alpha 
    return U, E, V, K


def kPCA_test(Xtest,K,E):
    K.expand(Xtest)
    X_new = np.dot(K.testMat,E)
    return X_new


if __name__ == '__main__':
    Train = np.loadtxt('dataset//monks_1_train.txt')
    X = Train[:,1:-1]
    Y = Train[:,0]
    
    Test = np.loadtxt('dataset//monks_1_test.txt')
    X_t = Test[:,1:-1]
    Y_t = Test[:,0]
    
    arg_best = 0
    acc_best = 0
    for arg in [10,100,1000]:

        kernel_dict = {'type':'RBF','gamma' : arg}
        (X, E, V, K1) = pca(X, kernel_dict, 10)
        
        X_t = kPCA_test(X_t,K1,V)
        
        #Norm
        NorX = Dataset.Normalization()
        NorX.fit(X)
        NorY = Dataset.Normalization()
        NorY.fit(Y)
        
        X_N = NorX.fT(X)
        
        Y_N = NorY.fT(Y)
        
        Y_N = Y_N * 2 -1
        
        
        X_N_t = NorX.fT(X_t)
        
        Y_N_t = NorY.fT(Y_t)
        
        Y_N_t = Y_N_t * 2 -1
        
        #LSSVM
        (alpha,b,K2) = LSSVM.LSSVM_CV(X_N,Y_N,'LINEAR',[0.001,0.005,0.01,0.05,0.1,0.5,1,4,10,25,100])
        Y_predict = LSSVM._LSSVMpredict(X_N_t,K2,alpha,b,Y_N)
    
        acc = LSSVM._compare(Y_N_t,np.array(Y_predict).T)
        
        if acc > acc_best:
            acc_best = acc
            arg_best = arg

    print(arg,acc)
    
'''
fig, (ax1, ax2) = subplots(1, 2)
ax1.scatter(data[:50, 0], data[:50, 1], c = 'r')
ax1.scatter(data[50:, 0], data[50:, 1], c = 'b')
ax2.scatter(trans[:50, 0], trans[:50, 1], c = 'r')
ax2.scatter(trans[50:, 0], trans[50:, 1], c = 'b')
show()
'''