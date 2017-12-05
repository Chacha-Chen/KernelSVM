# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:42:19 2017
Maybe not used for now.
@author: Dyt
"""


#linear kernel 有没有用到gammal参数？

import numpy

def kernel(x1,x2,k_type,gammaVal):
    #x1,x2 numpy.array
    if k_type == 'rbf':
        
        K = numpy.dot(x1,x2)
    return K