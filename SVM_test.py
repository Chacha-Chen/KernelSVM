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

X_N = NorX.forwardTransform()

Y_N = NorY.forwardTransform()

svm = Algorithms.SVM()

svm.Data.load(X_N,Y_N)

#svm.Data.set_up_for_SVM()
    

svm.clf()



svm.train(c=0.01,kernal='rbf')



svm.evaluate()

    
svm.predict()
    
