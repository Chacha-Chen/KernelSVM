# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:15:38 2018

@author: Administrator

KSVM module for indefinate kernels
"""

import numpy as np
import Kernel
import Algorithms

#对矩阵g对角化
def trans_mat(g):
    m,n = np.linalg.eig(g)
    r1 = 0
    r2 = m.shape[0] - 1
    e1 = np.zeros(m.shape[0])
    e2 = np.zeros(m.shape[0])
    v1 = np.zeros_like(g)
    v2 = np.zeros_like(g)
    for i in range(m.shape[0]):
        if m[i] > 0:
            e1[r1] = m[i]
            v1[:,r1] = n[:,i]
            r1 = r1 + 1
        else:
            e2[r2] = m[i]  
            v2[:,r2] = n[:,i]
            r2 = r2 - 1
    d1 = np.diag(e1)
    d2 = np.diag(e2)
    p1 = np.dot(np.dot(n,d1),n.T)
    p2 = np.dot(np.dot(n,d2),n.T)
            
    return (p1,p2)


def _KSVMtrain(X,Y,kernel_dict):
    m = Y.shape[0]

    if kernel_dict['type'] == 'RBF':
        K = Kernel.RBF(m,kernel_dict['gamma'])
        K.calculate(X)
    elif kernel_dict['type'] == 'LINEAR':
        K = Kernel.LINEAR(m)
        K.calculate(X)
    elif kernel_dict['type'] == 'POLY':
        K = Kernel.POLY(m)
        K.calculate(X)
    elif kernel_dict['type'] == 'TANH':
        K = Kernel.TANH(m,kernel_dict['c'],kernel_dict['d'])
        K.calculate(X)
    elif kernel_dict['type'] == 'TL1':
        K = Kernel.TL1(m,kernel_dict['rho'])
        K.calculate(X)
    
    p1,p2 = trans_mat(K.kernelMat)
    K.kernelMat = np.dot((p1 - p2),K.kernelMat)
    
    #根据SVM求出alpha,b   ？？？
    svm = Algorithms.SVM(X, Y, kernel_dict)
    
    #更新alpha
    alpha = np.dot((p1 - p2),svm.alphas)
    b = svm.b
    
    return (alpha,b,K)

def _KSVMpredict(Xtest,K,alpha,b):
    K.expand(Xtest)
    f = b + np.dot(K.testMat,alpha)
    Y_predict = f
    Y_predict[Y_predict >= 0] = 1
    Y_predict[Y_predict < 0] = -1
    
    return Y_predict


def _KSVMcompare(Ytest,Y_predict):
    #in np.array
    Error = (Ytest - Y_predict) / 2
    es = np.linalg.norm(Error,1)
    acc = 1 - es / Ytest.shape[0]
    return acc

def KSVM_CV(X,Y,kernel_type,arg1=None,arg2 = None):
    #RBF
    if kernel_type == 'RBF':
        A = [0] * 10
        B = [0] * 10
        
        indices = np.random.permutation(X.shape[0])   
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] 
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]

        best_gamma = None
        acc = 0
        acc_best = 0      
        for gammaVal in arg1:
            avg_acc = 0
            for i in range(10):
                X_test = A[i]
                Y_test = B[i]
                    

                X_train = np.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                Y_train = np.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                kernel_dict = {'type':'RBF', 'gamma' : gammaVal}
                (alpha,b,K)=_KSVMtrain(X_train,Y_train,kernel_dict)
                    
                Y_predict = _KSVMpredict(X_test,K,alpha,b)

                    
                    
                acc = _KSVMcompare(Y_test,Y_predict)
                    
                avg_acc = avg_acc +acc/10
                    
            if avg_acc > acc_best:
                acc_best = avg_acc

                best_gamma = gammaVal


        kernel_dict = {'type':'RBF', 'gamma' : best_gamma}
        (alpha,b,K)=_KSVMtrain(X,Y,kernel_dict)
        
        return (alpha,b,K)
    
     #Linear
    if kernel_type == 'LINEAR':
        A = [0] * 10
        B = [0] * 10
        
        indices = np.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]

        acc = 0
        acc_best = 0      
        
        avg_acc = 0
        for i in range(10):
            X_test = A[i]
            Y_test = B[i]
                    

            X_train = np.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
            Y_train = np.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
            kernel_dict = {'type':'LINEAR'}
            (alpha,b,K)=_KSVMtrain(X_train,Y_train,kernel_dict)
                    
            Y_predict = _KSVMpredict(X_test,K,alpha,b)

                    
                    
            acc = _KSVMcompare(Y_test,Y_predict)
                    
            avg_acc = avg_acc +acc/10
                    
            if avg_acc > acc_best:
                acc_best = avg_acc

                best_gamma = gammaVal


        kernel_dict = {'type':'LINEAR'}
        (alpha,b,K)=_KSVMtrain(X,Y,kernel_dict)
        
        return (alpha,b,K)

    #POLY
    if kernel_type == 'POLY':
        A = [0] * 10
        B = [0] * 10
        
        indices = np.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]

        best_c = None
        best_d= None
        acc = 0
        acc_best = 0      
        for c in arg1:
            for d in arg2:
                avg_acc = 0
                for i in range(10):
                    X_test = A[i]
                    Y_test = B[i]
                    

                    X_train = np.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                    Y_train = np.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                    kernel_dict = {'type':'POLY', 'c' : c, 'd' : d}
                    (alpha,b,K)=_KSVMtrain(X_train,Y_train,kernel_dict)
                    
                    Y_predict = _KSVMpredict(X_test,K,alpha,b)

                    
                    
                    acc = _KSVMcompare(Y_test,Y_predict)
                    
                    avg_acc = avg_acc +acc/10
                    
                if avg_acc > acc_best:
                    acc_best = avg_acc

                    best_c = c
                    best_d = d


        kernel_dict = {'type':'POLY', 'c' : best_c, 'd' : best_d}
        (alpha,b,K)=_KSVMtrain(X,Y,kernel_dict)
        
        return (alpha,b,K)

    #TANH
    if kernel_type == 'TANH':
        A = [0] * 10
        B = [0] * 10
        
        indices = np.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]

        best_c = None
        best_d= None
        acc = 0
        acc_best = 0      
        for c in arg1:
            for d in arg2:
                avg_acc = 0
                for i in range(10):
                    X_test = A[i]
                    Y_test = B[i]
                    

                    X_train = np.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                    Y_train = np.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                    kernel_dict = {'type':'TANH', 'c' : c, 'd' : d}
                    (alpha,b,K)=_KSVMtrain(X_train,Y_train,kernel_dict)
                    
                    Y_predict = _KSVMpredict(X_test,K,alpha,b)

                    
                    
                    acc = _KSVMcompare(Y_test,Y_predict)
                    
                    avg_acc = avg_acc +acc/10
                    
                    if avg_acc > acc_best:
                        acc_best = avg_acc

                        best_c = c
                        best_d = d


        kernel_dict = {'type':'TANH', 'c' : best_c, 'd' : best_d}
        (alpha,b,K)=_KSVMtrain(X,Y,kernel_dict)
        
        return (alpha,b,K)

    #TL1
    if kernel_type == 'TL1':
        A = [0] * 10
        B = [0] * 10
        
        indices = np.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = X[indices]  
        rand_data_y = Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]

        best_rho = None
        acc = 0
        acc_best = 0      
        for rho in arg1:
            avg_acc = 0
            for i in range(10):
                X_test = A[i]
                Y_test = B[i]
                    

                X_train = np.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                Y_train = np.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                kernel_dict = {'type':'TL1', 'rho' : rho}
                (alpha,b,K)=_KSVMtrain(X_train,Y_train,kernel_dict)
                    
                Y_predict = _KSVMpredict(X_test,K,alpha,b)

                    
                    
                acc = _KSVMcompare(Y_test,Y_predict)
                    
                avg_acc = avg_acc +acc/10
                    
            if avg_acc > acc_best:
                acc_best = avg_acc

                best_rho = rho


        kernel_dict = {'type':'TL1', 'rho' : best_rho}
        (alpha,b,K)=_KSVMtrain(X,Y,kernel_dict)
        
        return (alpha,b,K)

'''
#test code 
#k1 = np.array([[0,1.,1,-1],[1,0,-1,1],[1,-1,0,1],[-1,1,1,0]])   
k1 = np.array([[1,-2.,0],[-2,1,-2],[0,-2,1]])        
m,n = trans_mat(k1)
a,b = np.linalg.eig(k1)
k2 = np.dot((m - n),k1)
s,v = np.linalg.eig(k2)
'''