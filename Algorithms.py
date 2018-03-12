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
        #self.gamma_ = 1
        self.IK = IK


    def train(self,C=[0.01,1,10,100],tol=1e-3):
        
        m = self.Y.shape[0]
         
            
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
#        '''
#        X_num=self.X.shape[0]
#        train_index=range(X_num)
#        test_size=int(X_num*0.1)+1
#        for i in range(9):
#            test_index=[]
#            for j in range(test_size):
#                randomIndex=int(numpy.random.uniform(0,len(train_index)))
#                test_index.append(train_index[randomIndex])
#                #del train_index[randomIndex]
#            A[i]=self.X[test_index,:]
#            B[i]=self.Y[test_index,:]
#        A[9]=self.X.ix_[train_index]
#        B[9]=self.Y.ix_[train_index]
#        '''
        
        acc_best = 0
        C_best = None
        avg_acc = 0
#        gamma_best = None
        for CVal in C:
#            for gammaVal in gamma:
#                avg_acc = 0
            for i in range(10):
                X_test = A[i]
                Y_test = B[i]
                    

                    # X_train = None
                    # Y_train = None

                    #model= SMO.SMO_Model(X_train, Y_train, CVal,  kernel,gammaVal, tol=1e-3, eps=1e-3)
                    #output_model=SMO.SMO(model)

                    #根据output_model的参数信息计算对应decision_function----->推得accuracy
                    #acc = _evaulate(output_model)

                X_train = numpy.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                Y_train = numpy.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
#                SMO.GG = gammaVal
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
                    #更新C gamma
                    C_best = CVal
#                    gamma_best =gammaVal
#                    self.gamma = gamma_best


        #最后一遍train
#        SMO.GG = gamma_best
        
        #!K
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

                        # C_best = C
                        # gamma_best =gamma
						
        # (w,b) = SMO(X_train,Y_train,C_best,gamma_best,kernal,tol=1e-3)
        # self.w = w
        # self.b = b        
        return None

    def _SVMpredict(self,Xtest,K,alpha,b,Y):
        '''
        K.expand(Xtest)
        f = b + numpy.dot(K.testMat,alpha)
        Y_predict = f
        Y_predict[Y_predict >= 0] = 1
        Y_predict[Y_predict < 0] = -1
        
        return Y_predict
        '''
        K.expand(Xtest)
        A = np.multiply(alpha,Y)
    
        #f = b + np.dot(K.testMat,alpha)
        f = b + np.dot(K.testMat,A)
        #f = b + np.dot(K.testMat,np.multiply(alpha,Y))
        Y_predict = f
        Y_predict[Y_predict >= 0] = 1
        Y_predict[Y_predict < 0] = -1
        
        return Y_predict
    
    
    def evaluate(self,Ytest,Y_predict):
        #in np.array
        Error = (Ytest - Y_predict) / 2
        es = LA.norm(Error,1)
        acc = 1 - es / Ytest.shape[0]
        return acc

#    def evaluate(self,X_test,Y,):
#        
#        Y_predict = numpy.zeros(X_test.shape[0])
#        for i in range(X_test.shape[0]):
#            Y_predict[i] = SMO.decision_function(self.alphas, self.Y, K.kernelMat, self.X, X_test[i], self.b,self.gamma_)
#        error = Y - Y_predict
#        
#        mis = numpy.linalg.norm(error,0)
#        
#        acc = 1 - mis / Y.shape[0]
#        
#        return acc
    
    

#    
#'''
#    Single training process Using SMO
#    
#    def _train(self,x_train,y_train,C,kernalName,eps):
#        self.C = C
#        self.eps = eps
#        
#        self.kernalName = kernalName
#        self.kernal = Kernel.Kernel()
#        
#        #Lagrange Multiplier
#        self.alpha = numpy.zeros((self.Data.number_of_examples,2))
#        pass
#    
#    def set_parameters(self):
#        pass
#    
#    def _takestep(self,i1,i2):
#        if i1 == i2:
#            return 0
#        
#        y1 = self.Data.dataY[i1]
#        y2 = self.Data.dataY[i2]
#        
#        v1 = self._predict(self.Data.dataX[i1])
#        v2 = self._predict(self.Data.dataX[i2])
#        
#        if self.error_cache[i1]:
#            E1 = self.error_cache[i1]
#        else:
#            E1 = v1 - y1
#            
#        if self.error_cache[i2]:
#            E2 = self.error_cache[i2]
#        else:
#            E2 = v2 - y2
#            
#        s = y1 * y2
#        
#        #Compute L,H
#        if s == -1:
#            L = max(0,self.alpha[i2] - self.alpha[i1])
#            H = min(self.C,self.C + self.alpha[i2] - self.alpha[i1])
#        else:
#            L = max(0,self.alpha[i2] + self.alpha[i1] - self.C)
#            H = min(self.C,self.alpha[i2] + self.alpha[i1])
#            
#        if (L == H):
#            return 0
#        k11 = self.kernal.compute(self.Data.dataX[i1],self.Data.dataX[i1])
#        k12 = self.kernal.compute(self.Data.dataX[i1],self.Data.dataX[i2])
#        k22 = self.kernal.compute(self.Data.dataX[i2],self.Data.dataX[i2])
#        
#        eta = 2 * k12 - k11 - k22
#        if eta < 0:
#            a2 = self.alpha[i2] - y2 * (E1 - E2) / eta
#            if a2 < L:
#                a2 = L
#            elif a2 > H:
#                a2 = H
#        else:
#            Lobj = self._objective_function(i1,i2,self.alpha[i1],self.alpha[i2],L,s,y1,y2,k11,k12,k22,v1,v2)
#            Hobj = self._objective_function(i1,i2,self.alpha[i1],self.alpha[i2],H,s,y1,y2,k11,k12,k22,v1,v2)
#            if Lobj > Hobj + self.eps:
#                a2 = L
#            elif Lobj < Hobj - self.eps:
#                a2 = H
#            else:
#                a2 = self.alpha[i1]
#        if a2 < 1e-8:
#            a2 = 0
#        elif a2 > self.C - 1e-8:
#            a2 = self.C
#        
#        if abs(a2 - self.alpha[i2]) < self.eps * (a2 + self.alpha[i2] + self.eps):
#            return 0
#        
#        a1 = self.alpha[i1] + s * (self.alpha[i2] - a2)
#        
#        #Update threshold to reflect change in Lagrange multipliers
#        
#        #Update weight vector to reflect change in a1 & a2, if linear SVM
#        
#        #Update error cache using new Lagrange multipliers
#        
#        #Store a1 in the alpha array
#        self.alpha[i1] = a1
#        #Store a2 in the alpha array
#        self.alpha[i2] = a2
#        
#        
#        return 1
#            
#    def _objective_function(self,i1,i2,a1_old,a2_old,a2,s,y1,y2,k11,k12,k22,v1,v2):
#        #gamma = a1 + s * a2
#        gamma = a1_old + s * a2_old
#        
#        #我没有理解错v1 v2吧
#        #v1 = self._predict(self.Data.dataX[i1])
#        #v2 = self._predict(self.Data.dataX[i2])
#        
#        v1 = self._predict()
#        W = gamma -s * a2 + a2 - 0.5 * k11 * (gamma - s * a2) * (gamma - s * a2)
#        - 0.5 * k22 * a2 * a2 - s * k12 * (gamma - s * a2) * a2
#        - y1 * (gamma - s * a2) * v1 - y2 * a2 * v2
#        return W
# 
#
#    
#if __name__=="__main__":
#    svm = SVM()
#'''
