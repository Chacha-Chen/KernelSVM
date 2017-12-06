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