# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:42:19 2017
Maybe not used for now.
@author: Dyt
"""


#linear kernel 有没有用到gammal参数？

import math
import numpy as np
from numpy import linalg as LA
#import SMO
'''
def kernel_cal(x1,x2,k_type,gammaVal):
    #x1,x2 numpy.array
	
	num = x1.shape[0]
	for i in range(num):
        diff = x1[i, :] - x2
    K = exp(numpy.dot(diff,diff) / (-gammaVal)) 
    
	if k_type == 'rbf':        
        K = numpy.dot(x1,x2)
		
    return K
'''

# used in SMO algorithm
# input:2 vectors 
# ouput: a value
# change to input a matrix 
# output: a kernel matrix 

#def kernel_cal(x1,x2,k_type,gammaVal):
#    
#    diff = x1 - x2
#    K = np.exp(np.dot(diff,diff) * (-gammaVal)) 
#    #print(x1.shape)
#    return K

#  New Code For Kernel
class kernel:
    #samples is the number of samples
    def __init__(self,samples):
        '''
        Two Mat must be converted into np.array
        '''
        self.samples = samples
        self.kernelMat = np.zeros((samples,samples))
        self.testMat = None
        
    def call(self,i,j):
        return self.kernelMat[i][j]
    
    def _call_test(self,idx_test,idx_train):
        return self.testMat[idx_test][idx_train]

    
class RBF(kernel):
    def __init__(self,samples,gamma):
        kernel.__init__(self,samples)
        self.gamma = gamma;
    
    def calculate(self,X):
        X2 = np.sum(np.multiply(X, X), 1) # sum colums of the matrix
        K0 = np.matrix(X2) + np.matrix(X2).T - 2 * np.dot(X,X.T)
        self.kernelMat = np.array(np.power(np.exp(-1.0 / self.gamma**2), K0))
        self.X = X
        
        
    '''
    Update kernel for test and train at the same time
    '''
    def parameter_update_train(self,new_gamma):
        old_gamma = self.gamma
        self.gamma = new_gamma
        self.kernelMat = np.power(self.kernelMat,(old_gamma / new_gamma)**2)
        self.testMat = np.power(self.testMat,(old_gamma / new_gamma)**2)
    
    '''
    Calculate the kernel for test data
    '''
    def expand(self,Xtest):
        X2_train = np.sum(np.multiply(self.X,self.X),1)
        X2_test = np.sum(np.multiply(Xtest,Xtest),1)
        K0 = np.matrix(X2_train) + np.matrix(X2_test).T -2 * np.dot(Xtest,self.X.T)
        self.testMat = np.array(np.power(np.exp(-1.0 / self.gamma**2), K0))


class LINEAR(kernel):
    def __init__(self,samples):
        kernel.__init__(self,samples)
    
    def calculate(self,X):
        self.kernelMat = np.dot(X,X.T)
        self.X = X
    
    def expand(self,Xtest):
        self.testMat = np.dot(Xtest,self.X.T)
    

class POLY(kernel):
    #c>=0    d in N+
    def __init__(self,samples,c,d):
        kernel.__init__(self,samples)
        self.c = c;
        self.d = d;
    
    def calculate(self,X):
        self.kernelMat = np.power((np.dot(X,X.T) + self.c),self.d)
        self.X = X
        
    def expand(self,Xtest):
        self.testMat = np.power((np.dot(Xtest,self.X.T) + self.c),self.d)
    

class TANH(kernel):
    def __init__(self,samples,c,d):
        kernel.__init__(self,samples)
        self.c = c;
        self.d = d;
    
    def calculate(self,X):
        self.kernelMat = np.tanh(np.dot(X,X.T) + self.c)
        self.X = X
        
    def expand(self,Xtest):
        self.testMat = np.tanh(np.dot(Xtest,self.X.T) + self.c)

    
class TL1(kernel):
    def __init__(self,samples,rho):
        kernel.__init__(self,samples)
        self.rho = rho;
    
    def calculate(self,X):
        # Piecewise Calculation
        for i in range(self.samples):
            for j in range(self.samples):
                self.kernelMat[i][j] = self.rho - LA.norm((X[i]-X[j]),1)
        
        self.kernelMat[self.kernelMat<0] = 0
        self.X = X
        
    def expand(self,Xtest):
        for i in range(Xtest.shape[0]):
            for j in range(self.samples):
                self.testMat[i][j] = self.rho - LA.norm((Xtest[i]-self.X[j]),1)
        
        self.testMat[self.testMat<0] = 0
    

    
# Sample Code For Call
        
        
    
if __name__ == '__main__':
    X = np.ones((3,3))
    K = RBF(3,1)
    K.calculate(X)
    print(K.call(1,2))
    #K.parameter_update(10)
    Xtest = 2 * np.ones((2,3))
    K.expand(Xtest=Xtest)
    print(K._call_test(1,2))
    #For RBF kernel, parameter_update can increase efficiency in Cross-Validation.
    
    
    
X = np.array([[1,1,0],[1,2,3]])
y = np.array([[0,1,1],[3,4,5]])

gamma =1
X2 = np.sum(np.multiply(X, X), 1) # sum colums of the matrix
K0 = np.matrix(X2) + np.matrix(X2).T - 2 * np.dot(X,X.T)
kernelMat = np.array(np.power(np.exp(-1.0 / gamma**2), K0))
print(X,X2,K0,kernelMat)

#print(sum(np.multiply(x, x),1))


#print(np.dot(diff,diff))
#print(x.shape[0])
#print(kernel_cal(x,y,'rbf',1))
