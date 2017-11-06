# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:10:40 2017
算法模块

基类：监督学习机

子类：支持向量机


目前设想到这样的架构，觉得这样代码重用会比较好  把基础的很多功能做好以后再拓展其他
算法也比较容易

我觉得还可以找老师探讨一下，其他的架构，如果需要的话需要了解一下机器学习中主要算法之
间的关联，以便来设计更好的架构

@author: Dyt
"""


import  Model
import Dataset
import Kernel
import numpy

'''
监督学习类
有基本的功能
设想是各种算法都可以从这里继承，这里的代码可以重用
'''
class SupervisedLearning:
    
    '''
    初始化
    所有的监督学习均有数据集
    因此该父类拥有Dateset作为成员变量 并拥有
    后续子类无需再添加次成员
    
    将通过
    LearningMachine=SupervisedLearning()
    LearningMachine.Data.load()
    等方式进行使用
    '''
    def __init__(self):
        '''
        该类内部不需再写关于dataset处理的内容
        可以在外部直接调用Dataset的成员函数
        '''
        self.Data=Dataset.Dataset()
        self.Data.bool_supervised = True
        self.Data.dimension_y = 1
        pass
    
    
    '''
    训练核心算法
    可以内部调用也可以外部调用
    '''
    def train(self,x_trains,y_trains,parameters):
        pass
    
    
    '''
    训练完毕后，给定一组x，输出相应的结果
    '''
    def predict(self,x_predict):
        pass
    
    
    '''
    结果评估
    给定一组测试集，输出准确率等信息
    需要利用predict
    '''
    def evaluate(self,x_test,y_test):
        pass
    
    
    '''
    交叉检验
    包含 对数据集的分割  需要利用train 和 evaluate
    '''
    def cross_validation(self,x_train,y_train):
        pass
    
    
    #保存模型
    def save(self):
        pass
    
    #夹在模型
    def load(self):
        pass


class Regression(SupervisedLearning):
    def __init__(self):
        pass
    
class Classification(SupervisedLearning):
    def __init__(self):
        pass
    

'''
从基类继承 可以继承evaluate等相同功能的函数
SVM特有的（如训练算法）在这里重载
'''
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
    
    
    
if __name__=="__main__":
    svm = SVM()
    