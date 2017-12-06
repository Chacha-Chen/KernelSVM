# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:23:45 2017

@author: Dyt
"""
import Algorithms
import numpy
import Dataset


Train = numpy.loadtxt('monks_1_train.txt')

X = Train[:,:-1]

Y = Train[:,-1]




NorX = Dataset.Normalization()
NorX.fit(X)

NorY = Dataset.Normalization()
NorY.fit(Y)

X_N = NorX.fT(X)

Y_N = NorY.fT(Y)


#Initializing SVM
svm = Algorithms.SVM(X_N, Y_N, 'rbf')


#svm.Data.set_up_for_SVM()
    

svm.train(C=0.01, gamma=0.01, kernel='rbf')













svm.evaluate()

svm.predict()
    
