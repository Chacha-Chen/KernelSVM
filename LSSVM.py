# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:49:48 2018

@author: Dynasting

LSSVM module for indefinate kernels
"""

import numpy as np
import numpy
from numpy import linalg as LA
import Kernel

'''
_LSSVMtrain function without CV
kernel_dict = {'type':'RBF', 'gamma' : 1}
(alpha,b,K)=_LSSVMtrain(X,Y,kernel_dict,regulator)
 Y np.array
'''
def _LSSVMtrain(X,Y,kernel_dict,regulator):
    m = Y.shape[0]
    # Kernel
    if kernel_dict['type'] == 'RBF':
        K = Kernel.RBF(m,kernel_dict['gamma'])
        K.calculate(X)
    elif kernel_dict['type'] == 'LINEAR':
        K = Kernel.LINEAR(m)
        K.calculate(X)
    elif kernel_dict['type'] == 'POLY':
        K = Kernel.POLY(m,kernel_dict['c'],kernel_dict['d'])
        K.calculate(X)
    elif kernel_dict['type'] == 'TANH':
        K = Kernel.TANH(m,kernel_dict['c'],kernel_dict['d'])
        K.calculate(X)
    elif kernel_dict['type'] == 'TL1':
        K = Kernel.TL1(m,kernel_dict['rho'])
        K.calculate(X)
    
    H = np.multiply(np.dot(np.matrix(Y).T,np.matrix(Y)),K.kernelMat)
    M_BR = H + np.eye(m) / regulator
    #Concatenate
    L_L = np.concatenate((np.matrix(0),np.matrix(Y).T),axis = 0)
    L_R = np.concatenate((np.matrix(Y),M_BR),axis = 0)
    L = np.concatenate((L_L,L_R),axis = 1)
    R = np.ones(m+1)
    R[0] = 0
    
    #solve
    b_a = LA.solve(L,R)
    b = b_a[0]
    alpha = b_a[1:]
    
    #return
    return (alpha,b,K)

def _LSSVMpredict(Xtest,K,alpha,b,Y):
    K.expand(Xtest)
    A = np.multiply(alpha,Y)

    #f = b + np.dot(K.testMat,alpha)
    f = b + np.dot(K.testMat,A)
    #f = b + np.dot(K.testMat,np.multiply(alpha,Y))
    Y_predict = f
    Y_predict[Y_predict >= 0] = 1
    Y_predict[Y_predict < 0] = -1
    
    return Y_predict

def _compare(Ytest,Y_predict):
    #in np.array
    Error = (Ytest - Y_predict) / 2
    es = LA.norm(Error,1)
    acc = 1 - es / Ytest.shape[0]
    return acc


# arg List
def LSSVM_CV(X,Y,kernel_type,GList,arg1=None,arg2 = None):
    #RBF
    if kernel_type == 'RBF':
        A = [0] * 10
        B = [0] * 10
        
        indices = numpy.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]
        
        best_G = None
        best_gamma = None
        acc = 0
        acc_best = 0
        for G in GList:
            for gammaVal in arg1:
                avg_acc = 0
                for i in range(10):
                    X_test = A[i]
                    Y_test = B[i]
                    

                    X_train = numpy.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                    Y_train = numpy.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                    kernel_dict = {'type':'RBF', 'gamma' : gammaVal}
                    (alpha,b,K)=_LSSVMtrain(X_train,Y_train,kernel_dict,G)
                    
                    Y_predict = _LSSVMpredict(X_test,K,alpha,b,Y_train)

                    
                    
                    acc = _compare(Y_test,Y_predict)
                    
                    avg_acc = avg_acc +acc/10
                    
                if avg_acc > acc_best:
                    acc_best = avg_acc

                    best_gamma = gammaVal
                    best_G =G



        #最后一遍train
        kernel_dict = {'type':'RBF', 'gamma' : best_gamma}
        (alpha,b,K)=_LSSVMtrain(X,Y,kernel_dict,best_G)
        
        return (alpha,b,K)
    
    #Linear
    if kernel_type == 'LINEAR':
        A = [0] * 10
        B = [0] * 10
        
        indices = numpy.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]
        
        best_G = None
        
        acc = 0
        acc_best = 0
        for G in GList:
            
            avg_acc = 0
            for i in range(10):
                X_test = A[i]
                Y_test = B[i]

                X_train = numpy.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                Y_train = numpy.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                kernel_dict = {'type':'LINEAR'}
                (alpha,b,K)=_LSSVMtrain(X_train,Y_train,kernel_dict,G)
                    
                Y_predict = _LSSVMpredict(X_test,K,alpha,b,Y_train)
                  
                acc = _compare(Y_test,Y_predict)
                    
                avg_acc = avg_acc +acc/10
                    
            if avg_acc > acc_best:
                acc_best = avg_acc
                best_G =G



        #最后一遍train
        kernel_dict = {'type':'LINEAR'}
        (alpha,b,K)=_LSSVMtrain(X,Y,kernel_dict,best_G)
        
        return (alpha,b,K)
    
    #POLY
    if kernel_type == 'POLY':
        A = [0] * 10
        B = [0] * 10
        
        indices = numpy.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]
        
        best_G = None
        best_c = None
        best_d= None
        acc = 0
        acc_best = 0
        for G in GList:
            for c in arg1:
                for d in arg2:
                    avg_acc = 0
                    for i in range(10):
                        X_test = A[i]
                        Y_test = B[i]
                    


                        X_train = numpy.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                        Y_train = numpy.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                        kernel_dict = {'type':'POLY', 'c' : c, 'd' : d}
                        (alpha,b,K)=_LSSVMtrain(X_train,Y_train,kernel_dict,G)
                    
                        Y_predict = _LSSVMpredict(X_test,K,alpha,b,Y_train)

                    
                    
                        acc = _compare(Y_test,Y_predict)
                    
                        avg_acc = avg_acc +acc/10
                    
                    if avg_acc > acc_best:
                        acc_best = avg_acc

                        best_c = c
                        best_d = d
                        best_G =G



        #最后一遍train
        kernel_dict = {'type':'POLY', 'c' : best_c, 'd' : best_d}
        (alpha,b,K)=_LSSVMtrain(X,Y,kernel_dict,best_G)
        
        return (alpha,b,K)
    
    #TANH
    if kernel_type == 'TANH':
        A = [0] * 10
        B = [0] * 10
        
        indices = numpy.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]
        
        best_G = None
        best_c = None
        best_d= None
        acc = 0
        acc_best = 0
        for G in GList:
            for c in arg1:
                for d in arg2:
                    avg_acc = 0
                    for i in range(10):
                        X_test = A[i]
                        Y_test = B[i]
                    


                        X_train = numpy.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                        Y_train = numpy.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                        kernel_dict = {'type':'TANH', 'c' : c, 'd' : d}
                        (alpha,b,K)=_LSSVMtrain(X_train,Y_train,kernel_dict,G)
                    
                        Y_predict = _LSSVMpredict(X_test,K,alpha,b,Y_train)

                    
                    
                        acc = _compare(Y_test,Y_predict)
                    
                        avg_acc = avg_acc +acc/10
                    
                    if avg_acc > acc_best:
                        acc_best = avg_acc

                        best_c = c
                        best_d = d
                        best_G =G



        #最后一遍train
        kernel_dict = {'type':'TANH', 'c' : best_c, 'd' : best_d}
        (alpha,b,K)=_LSSVMtrain(X,Y,kernel_dict,best_G)
        
        return (alpha,b,K)
    
    #TL1
    if kernel_type == 'TL1':
        A = [0] * 10
        B = [0] * 10
        
        indices = numpy.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]
        
        best_G = None
        best_rho = None
        acc = 0
        acc_best = 0
        for G in GList:
            for rho in arg1:
                avg_acc = 0
                for i in range(10):
                    X_test = A[i]
                    Y_test = B[i]
                    

                    X_train = numpy.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                    Y_train = numpy.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                    kernel_dict = {'type':'TL1', 'rho' : rho}
                    (alpha,b,K)=_LSSVMtrain(X_train,Y_train,kernel_dict,G)
                    
                    Y_predict = _LSSVMpredict(X_test,K,alpha,b,Y_train)

                    
                    
                    acc = _compare(Y_test,Y_predict)
                    
                    avg_acc = avg_acc +acc/10
                    
                if avg_acc > acc_best:
                    acc_best = avg_acc

                    best_rho = rho
                    best_G =G



        #最后一遍train
        kernel_dict = {'type':'TL1', 'rho' : best_rho}
        (alpha,b,K)=_LSSVMtrain(X,Y,kernel_dict,best_G)
        
        return (alpha,b,K)


# Test Code for _LSSVMtrain

if __name__ == '__main__':
    X = np.random.rand(5,8) - 0.5
    Y = np.array([1,-1,-1,-1,-1])
    kernel_dict = {'type':'RBF', 'gamma' : 1}
    (alpha,b,K)=_LSSVMtrain(X,Y,kernel_dict,1)
    Xtest = np.random.rand(3,8) - 0.5
    Ytest = np.random.rand(3)
    Y_predict = _LSSVMpredict(Xtest,K,alpha,b)
    
    LSSVM_CV(X,Y,'RBF',[0.1,1,10],[0.1,1,10],arg2 = None)


