# -*- coding: utf-8 -*-

import numpy


class Normalization():
    def __init__(self):
          self.maxVector = None
          self.minVector = None
    '''
    input_data  numpy.array
    '''
    def fit(self,input_data):
        #check input_data type
        maxVector = input_data.max(axis=0)
        minVector = input_data.min(axis=0)
        
        self.maxVector = maxVector
        self.minVector = minVector
        
        
    def fT(self,input_data):
            
        up = input_data-self.minVector
        down = (self.maxVector - self.minVector)
        
        normalizedData = numpy.divide(up,down)
            
        return normalizedData
            
if __name__== '__main__':
    
    X = numpy.array([[2.0,3.0,3.0],[1.5,2.0,8.0],[5,7,78]])

    Y = numpy.array([[1.0],[2],[8]])
    
    NorX = Normalization()
    
    
    NorX.fit(X)
    
    
    NorY = Normalization()
    NorY.fit(Y)
    
    X_N = NorX.fT(X)
    
    Y_N = NorY.fT(Y)
    
            
            