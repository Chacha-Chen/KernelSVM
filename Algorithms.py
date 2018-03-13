# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:10:40 2017
SVM算法模块
@author: Dyt
initialized with training set and kernel type
"""


import SMO
import numpy
import numpy as np
from numpy import linalg as LA
import Kernel


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



class SVM():

    #using kernel_dict to save the kernel type and gamma value??
    #kernel_dict = {'type':'RBF', 'gamma' : 1} 一个参数  循环放到cross_validation里面
    #参数C是一个向量
    
    
    def __init__(self, x_train, y_train, kernel_dict, IK=False):

        self.X = x_train  # training data，m*n
        self.Y = y_train  # class label vector，1*m
        self.kernel_dict = kernel_dict  # kernel_dict = {'type':'RBF', 'gamma' : 1}
        self.alphas = numpy.zeros(len(self.X))  # lagrange multiplier vector, initialized as zeros
        self.b = None  # scalar bias term
        self.IK = IK


    def train(self,C=[0.01,1,10,100],tol=1e-3):
        
         
            
        A = [0] * 10
        B = [0] * 10
        
        indices = numpy.random.permutation(self.X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量  
        rand_data_x = self.X[indices]  
        rand_data_y = self.Y[indices] # data_y就是标记（label）  
        
        l = int(len(indices) / 10) 
        
        for i in range(9):
            A[i] = rand_data_x[i*l:i*l+l]
            B[i] = rand_data_y[i*l:i*l+l]
        
        A[9] = rand_data_x[9*l:]
        B[9] = rand_data_y[9*l:]
        
        acc_best = 0
        C_best = None
        avg_acc = 0
        for CVal in C:
            for i in range(10):
                X_test = A[i]
                Y_test = B[i]
                    



                #根据output_model的参数信息计算对应decision_function----->推得accuracy

                X_train = numpy.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                Y_train = numpy.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                # calculate Kernel Matrix then pass it to SMO.
                if self.IK:
                    if self.kernel_dict['type'] == 'TANH':
                        K = Kernel.TANH(X_train.shape[0],self.kernel_dict['c'],self.kernel_dict['d'])
                        K.calculate(X_train)
                    elif self.kernel_dict['type'] == 'TL1':
                        K = Kernel.TL1(X_train.shape[0],self.kernel_dict['rho'])
                        K.calculate(X_train)
                    
                    p1,p2 = trans_mat(K.kernelMat)
                    K.kernelMat = np.dot((p1 - p2),K.kernelMat)
                
                if self.kernel_dict['type'] == 'RBF':
                    K = Kernel.RBF(X_train.shape[0],self.kernel_dict['gamma'])
                    K.calculate(X_train)
                elif self.kernel_dict['type'] == 'LINEAR':
                    K = Kernel.LINEAR(X_train.shape[0])
                    K.calculate(X_train)
                elif self.kernel_dict['type'] == 'POLY':
                    K = Kernel.POLY(X_train.shape[0],self.kernel_dict['c'],self.kernel_dict['d'])
                    K.calculate(X_train)
                elif self.kernel_dict['type'] == 'TANH':
                    K = Kernel.TANH(X_train.shape[0],self.kernel_dict['c'],self.kernel_dict['d'])
                    K.calculate(X_train)
                elif self.kernel_dict['type'] == 'TL1':
                    K = Kernel.TL1(X_train.shape[0],self.kernel_dict['rho'])
                    K.calculate(X_train)
           
                model= SMO.SMO_Model(X_train, Y_train, CVal, K, tol=1e-3, eps=1e-3)

                output_model=SMO.SMO(model)
                
                #IK
                if self.IK:
                    output_model.alphas = np.dot((p1 - p2),output_model.alphas)
                    
                acc = SMO._evaluate(output_model,X_test,Y_test)
                    
                avg_acc = avg_acc +acc/10
                    
                if avg_acc > acc_best:
                    acc_best = avg_acc
                    C_best = CVal

        #最后一遍train
        
        if self.IK:
            if self.kernel_dict['type'] == 'TANH':
                K = Kernel.TANH(self.X.shape[0],self.kernel_dict['c'],self.kernel_dict['d'])
                K.calculate(self.X)
            elif self.kernel_dict['type'] == 'TL1':
                K = Kernel.TL1(self.X.shape[0],self.kernel_dict['rho'])
                K.calculate(self.X)
                    
            p1,p2 = trans_mat(K.kernelMat)
            K.kernelMat = np.dot((p1 - p2),K.kernelMat)
        
        
        if self.kernel_dict['type'] == 'RBF':
            K = Kernel.RBF(self.X.shape[0],self.kernel_dict['gamma'])
            K.calculate(self.X)
        elif self.kernel_dict['type'] == 'LINEAR':
            K = Kernel.LINEAR(self.X.shape[0])
            K.calculate(self.X)
        elif self.kernel_dict['type'] == 'POLY':
            K = Kernel.POLY(self.X.shape[0],self.kernel_dict['c'],self.kernel_dict['d'])
            K.calculate(self.X)
        elif self.kernel_dict['type'] == 'TANH':
            K = Kernel.TANH(self.X.shape[0],self.kernel_dict['c'],self.kernel_dict['d'])
            K.calculate(self.X)
        elif self.kernel_dict['type'] == 'TL1':
            K = Kernel.TL1(self.X.shape[0],self.kernel_dict['rho'])
            K.calculate(self.X)
        
        
        SVM_model = SMO.SMO(SMO.SMO_Model(self.X, self.Y , C_best, K, tol=1e-3, eps=1e-3))
        # 参数传递给最后生成的SVM类
        
        if self.IK:
            SVM_model.alphas = np.dot((p1 - p2),SVM_model.alphas)
        
        self.X = SVM_model.X
        self.Y = SVM_model.y
        self.kernel_dict = SVM_model.kernel
        self.alphas = SVM_model.alphas
        self.b = SVM_model.b

        return None

    def _SVMpredict(self,Xtest,K,alpha,b,Y):
        K.expand(Xtest)
        A = np.multiply(alpha,Y)
    
        f = b + np.dot(K.testMat,A)
        Y_predict = f
        Y_predict[Y_predict >= 0] = 1
        Y_predict[Y_predict < 0] = -1
        
        return Y_predict
    
    
    def evaluate(self,Ytest,Y_predict):
        Error = (Ytest - Y_predict) / 2
        es = LA.norm(Error,1)
        acc = 1 - es / Ytest.shape[0]
        return acc

