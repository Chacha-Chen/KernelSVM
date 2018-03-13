# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:23:45 2017

@author: Dyt
"""
import Algorithms
import numpy
import Dataset


Train = numpy.loadtxt('dataset//monks_1_train.txt')


X = Train[:,1:-1]
Y = Train[:,0]


NorX = Dataset.Normalization()
NorX.fit(X)

NorY = Dataset.Normalization()
NorY.fit(Y)

X_N = NorX.fT(X)
Y_N = NorY.fT(Y)
Y_N = Y_N * 2 -1



Test = numpy.loadtxt('dataset//monks_1_test.txt')
X_t = Test[:,1:-1]
Y_t = Test[:,0]



X_N_t = NorX.fT(X_t)
Y_N_t = NorY.fT(Y_t)
Y_N_t = Y_N_t * 2 -1


best_arg = 0
best_acc = 0
for arg in [0.1,1,2,5]:
    for arg2 in [1,2,3]:
        kernel_dict = {'type':'POLY', 'c' : arg, 'd':arg2}
        #kernel_dict = {'type':'TL1', 'rho' : arg}
        #Initializing SVM
        svm = Algorithms.SVM(X_N, Y_N, kernel_dict)
        
        #svm.Data.set_up_for_SVM()
        #svm.train(C=[0.01,1,10,100], gamma=[0.1,0.2,0.5,1.0], kernel='rbf')
        
        svm.train(C=[0.1,1,10])
        
        
        #_SVMpredict(Xtest,K,alpha,b,Y)
        Y_predict = svm._SVMpredict(X_N_t, svm.kernel_dict, svm.alphas, svm.b, Y_N)
        acc = svm.evaluate(Y_predict,Y_N_t)
        print(acc)
        if acc > best_acc:
            best_acc = acc
            best_arg = arg

print("bestacc",best_acc)
#svm.predict()
    
