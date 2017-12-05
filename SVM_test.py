# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:23:45 2017

@author: Dyt
"""
import Algorithms
import numpy
import Dataset



X = numpy.array([[1,3,3],[1,2,3]])

Y = numpy.array([[1],[2]])

NorX = Dataset.Normalization()


NorX.fit(X)


NorY = Dataset.Normalization()
NorY.fit(Y)

X_N = NorX.fT(X)

Y_N = NorY.fT(Y)


svm = Algorithms.SVM()



#svm.Data.set_up_for_SVM()
    





svm.train(X_N,Y_N,c=0.01,gamma=0.01,kernal='rbf')



svm.evaluate()

    
svm.predict()
    
