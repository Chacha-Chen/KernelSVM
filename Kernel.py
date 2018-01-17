# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:42:19 2017
Maybe not used for now.
@author: Dyt
"""


#linear kernel 有没有用到gammal参数？

import math
import numpy
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

def kernel_cal(x1,x2,k_type,gammaVal):
    
    
    diff = x1 - x2
    K = math.exp(numpy.dot(diff,diff) / (-gammaVal)) 
    #print(x1.shape)
    return K

class kernel:
    
    def __init__(self,size):
        self.kernelMat = numpy.zeros(size)
        
    def call(self,i,j):
        return self.kernelMat[i][j]

    
class RBF(kernel):
    def __init__(self,size,gamma):
        kernel.__init__(size)
        self.gamma = gamma;
    
    def calculate(self,X):
        pass


class LINEAR(kernel):
    def __init__(self,size):
        kernel.__init__(size)
    
    def calculate(self,X):
        pass

class POLY(kernel):
    def __init__(self,size,c,d):
        kernel.__init__(size)
        self.c = c;
        self.d = d;
    
    def calculate(self,X):
        pass

class TANH(kernel):
    def __init__(self,size,c,d):
        kernel.__init__(size)
        self.c = c;
        self.d = d;
    
    def calculate(self,X):
        pass

    
class TL1(kernel):
    def __init__(self,size,rho):
        kernel.__init__(size)
        self.rho = rho;
    
    def calculate(self,X):
        pass
    

    
    
    
if __name__ == '__main__':
    X = numpy.ones((3,3))
    K = TL1(X.shape,10)
    K.calculate(X)
    K.call(1,2)
    