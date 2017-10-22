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
        self.model = Model.LinearModel()
        pass
    
    def train(self,x_train,y_train):
        pass
    
    def set_parameters(self):
        pass
    
    
if __name__=="__main__":
    svm = SVM()
    