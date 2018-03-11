# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:39:12 2018

@author: Dynasting
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:23:45 2017

@author: Dyt
"""
import LSSVM
import numpy
import Dataset


Train = numpy.loadtxt('dataset///monks_2_train.txt')

X = Train[:,1:-1]

Y = Train[:,0]




NorX = Dataset.Normalization()
NorX.fit(X)

NorY = Dataset.Normalization()
NorY.fit(Y)

X_N = NorX.fT(X)

Y_N = NorY.fT(Y)

Y_N = Y_N * 2 -1



Test = numpy.loadtxt('dataset///monks_2_test.txt')

X_t = Test[:,1:-1]

Y_t = Test[:,0]






X_N_t = NorX.fT(X_t)

Y_N_t = NorY.fT(Y_t)

Y_N_t = Y_N_t * 2 -1



(alpha,b,K) = LSSVM.LSSVM_CV(X_N,Y_N,'RBF',[0.01,0.1,0.5,1,2,10,25,50,100],[0.001,0.01,0.1,0.2,0.5,1,5,10,40,100],arg2 = None)
#(alpha,b,K) = LSSVM.LSSVM_CV(X_N,Y_N,'LINEAR',[0.001,0.005,0.01,0.05,0.1,0.5,1,4,10,25,100])
#(alpha,b,K) = LSSVM.LSSVM_CV(X_N,Y_N,'POLY',[0.01,0.1,0.5,1,2,10,25,50,100],[0.001,0.01,0.1,0.2,0.5,1,5,10,40,100],[1,2,3,4,5])
#(alpha,b,K) = LSSVM.LSSVM_CV(X_N,Y_N,'TANH',[0.01,0.1,0.5,1,2,10,25,50,100],[-10,-5,-3,-2,-1,-0.5,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.5,1,3,5,10],[0.1,1,2,3,10])
#(alpha,b,K) = LSSVM.LSSVM_CV(X_N,Y_N,'TL1',[0.001,0.005,0.01,0.03,0.1,0.5,1,2,10,25,50,100],[0.001,0.005,0.01,0.05,0.1,0.2,0.5,1,2,5,10,20])
Y_predict = LSSVM._LSSVMpredict(X_N_t,K,alpha,b,Y_N)

acc = LSSVM._compare(Y_N_t,Y_predict)

print(acc)