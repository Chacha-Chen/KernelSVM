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
    p1 = np.dot(np.dot(np.dot(n,d1),n.T),np.linalg.inv(g))
    p2 = np.dot(np.dot(np.dot(n,d2),n.T),np.linalg.inv(g))
            
    return (p1,p2)


def _KSVMtrain(X,Y,kernel_dict,regulator):
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
    
    p1,p2 = trans_mat(K)
    KK = np.dot((p1 - p2),K)
    
    #根据SVM求出alpha,b   ？？？
    svm = Algorithms.SVM(X, Y, 'type')
    
    #更新alpha
    alpha_new = np.dot((p1 - p2),svm.alpha)
    
    return (alpha_new,svm.b,KK)

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


#test code 
#k1 = np.array([[0,1.,1,-1],[1,0,-1,1],[1,-1,0,1],[-1,1,1,0]])   
k1 = np.array([[1,-2.,0],[-2,1,-2],[0,-2,1]])        
m,n = trans_mat(k1)
a,b = np.linalg.eig(k1)
k2 = np.dot((m - n),k1)
s,v = np.linalg.eig(k2)

