# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:10:40 2017
SMO算法模块

基类：监督学习机

子类：支持向量机



@author: Chacha
"""

class SVM(Classification):
    def __init__(self):
        Classification.__init__(self)
        
        
        #SVM parameters
        self.input_size = None
        
        self.error_cache = numpy.zeros((self.Data.number_of_examples, 1))
        self.error_cache = self.error_cache + numpy.nan
        
        #self.target = numpy.zeros((self.Data.number_of_examples))
    def _predict(self,x_predict):
        pass
    
    def train(self,x_train,y_train,C,kernalName,eps):
        self.C = C
        self.eps = eps
        
        self.kernalName = kernalName
        self.kernal = Kernel.Kernel()
        
        #Lagrange Multiplier
        self.alpha = numpy.zeros((self.Data.number_of_examples,2))
        pass
    
    def set_parameters(self):
        pass
    
    def _takestep(self,i1,i2):
        if i1 == i2:
            return 0
        
        y1 = self.Data.dataY[i1]
        y2 = self.Data.dataY[i2]
        
        v1 = self._predict(self.Data.dataX[i1])
        v2 = self._predict(self.Data.dataX[i2])
        
        if self.error_cache[i1]:
            E1 = self.error_cache[i1]
        else:
            E1 = v1 - y1
            
        if self.error_cache[i2]:
            E2 = self.error_cache[i2]
        else:
            E2 = v2 - y2
            
        s = y1 * y2
        
        #Compute L,H
        if s == -1:
            L = max(0,self.alpha[i2] - self.alpha[i1])
            H = min(self.C,self.C + self.alpha[i2] - self.alpha[i1])
        else:
            L = max(0,self.alpha[i2] + self.alpha[i1] - self.C)
            H = min(self.C,self.alpha[i2] + self.alpha[i1])
            
        if (L == H):
            return 0
        k11 = self.kernal.compute(self.Data.dataX[i1],self.Data.dataX[i1])
        k12 = self.kernal.compute(self.Data.dataX[i1],self.Data.dataX[i2])
        k22 = self.kernal.compute(self.Data.dataX[i2],self.Data.dataX[i2])
        
        eta = 2 * k12 - k11 - k22
        if eta < 0:
            a2 = self.alpha[i2] - y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            Lobj = self._objective_function(i1,i2,self.alpha[i1],self.alpha[i2],L,s,y1,y2,k11,k12,k22,v1,v2)
            Hobj = self._objective_function(i1,i2,self.alpha[i1],self.alpha[i2],H,s,y1,y2,k11,k12,k22,v1,v2)
            if Lobj > Hobj + self.eps:
                a2 = L
            elif Lobj < Hobj - self.eps:
                a2 = H
            else:
                a2 = self.alpha[i1]
        if a2 < 1e-8:
            a2 = 0
        elif a2 > self.C - 1e-8:
            a2 = self.C
        
        if abs(a2 - self.alpha[i2]) < self.eps * (a2 + self.alpha[i2] + self.eps):
            return 0
        
        a1 = self.alpha[i1] + s * (self.alpha[i2] - a2)
        
        #Update threshold to reflect change in Lagrange multipliers
        
        #Update weight vector to reflect change in a1 & a2, if linear SVM
        
        #Update error cache using new Lagrange multipliers
        
        #Store a1 in the alpha array
        self.alpha[i1] = a1
        #Store a2 in the alpha array
        self.alpha[i2] = a2
        
        
        return 1
            
    def _objective_function(self,i1,i2,a1_old,a2_old,a2,s,y1,y2,k11,k12,k22,v1,v2):
        #gamma = a1 + s * a2
        gamma = a1_old + s * a2_old
        
        #我没有理解错v1 v2吧
        #v1 = self._predict(self.Data.dataX[i1])
        #v2 = self._predict(self.Data.dataX[i2])
        
        v1 = self._predict()
        W = gamma -s * a2 + a2 - 0.5 * k11 * (gamma - s * a2) * (gamma - s * a2)
        - 0.5 * k22 * a2 * a2 - s * k12 * (gamma - s * a2) * a2
        - y1 * (gamma - s * a2) * v1 - y2 * a2 * v2
        return W
    
    
