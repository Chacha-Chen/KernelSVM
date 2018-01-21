# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:49:48 2018

@author: Dynasting

LSSVM module for indefinate kernels
"""

import numpy as np
from numpy import linalg as LA
import Kernel

'''
_LSSVMtrain function without CV
kernel_dict = {'type':'RBF', 'gamma' : 1}
(alpha,b)=_LSSVMtrain(X,Y,kernel_dict,regulator)
 Y np.array
'''
def _LSSVMtrain(X,Y,kernel_dict,regulator):
    m = Y.shape[0]
    # Kernel
    if kernel_dict['type'] == 'RBF':
        K = Kernel.RBF(m,kernel_dict['gamma'])
        K.calculate(X)
    elif kernel_dict['type'] == 'LINEAR':
        K = Kernel.LINEAR(m)
        K.calculate(X)
    elif kernel_dict['type'] == 'POLY':
        K = Kernel.POLY(m)
        K.calculate(X)
    elif kernel_dict['type'] == 'TANH':
        K = Kernel.TANH(m,kernel_dict['c'],kernel_dict['d'])
        K.calculate(X)
    elif kernel_dict['type'] == 'TL1':
        K = Kernel.TL1(m,kernel_dict['rho'])
        K.calculate(X)
    
    H = np.multiply(np.dot(np.matrix(Y).T,np.matrix(Y)),K.kernelMat)
    M_BR = H + np.eye(m) / regulator
    #Concatenate
    L_L = np.concatenate((np.matrix(0),np.matrix(Y).T),axis = 0)
    L_R = np.concatenate((np.matrix(Y),M_BR),axis = 0)
    L = np.concatenate((L_L,L_R),axis = 1)
    R = np.ones(m+1)
    R[0] = 0
    
    #solve
    b_a = LA.solve(L,R)
    b = b_a[0]
    alpha = b_a[1:]
    
    #return
    return (alpha,b)

'''
# Test Code for _LSSVMtrain

if __name__ == '__main__':
    X = np.random.rand(5,5)
    Y = np.array([1,1,-1,-1,1])
    kernel_dict = {'type':'RBF', 'gamma' : 1}
    (alpha,b)=_LSSVMtrain(X,Y,kernel_dict,1)

'''

