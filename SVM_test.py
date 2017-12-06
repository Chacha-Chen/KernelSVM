# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:23:45 2017

@author: Dyt
"""
import Algorithms
import numpy
import Dataset


Train = numpy.loadtxt('monks_1_train.txt')

X = Train[:,1:-1]

Y = Train[:,0]




NorX = Dataset.Normalization()
NorX.fit(X)

NorY = Dataset.Normalization()
NorY.fit(Y)

X_N = NorX.fT(X)

Y_N = NorY.fT(Y)

Y_N = Y_N * 2 -1



Test = numpy.loadtxt('monks_1_test.txt')

X_t = Test[:,1:-1]

Y_t = Test[:,0]




NorX_t = Dataset.Normalization()
NorX_t.fit(X_t)

NorY_t = Dataset.Normalization()
NorY_t.fit(Y_t)

X_N_t = NorX.fT(X_t)

Y_N_t = NorY.fT(Y_t)

Y_N_t = Y_N_t * 2 -1


#Initializing SVM
svm = Algorithms.SVM(X_N, Y_N, 'rbf')


#svm.Data.set_up_for_SVM()
    


#svm.train(C=[0.01,1,10,100], gamma=[0.1,0.2,0.5,1.0], kernel='rbf')

svm.train(C=[0.01,0.1], gamma=[0.1,1], kernel='rbf')












print(svm.evaluate(X_N_t,Y_N_t))

#svm.predict()
    
