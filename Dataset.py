# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:42:12 2017

@author: Dyt

数据集处理模块

这一部分还没有设想好怎么来架构

numpy这个库对数组大小是有限制的
如果数据集过大  我们可能需要对数据集做特殊的处理
归一化等操作也可以放在这个模块中

我的想法是各个模块能和其他模块间不要耦合在一起影响思路
个人感觉这里想要完全和Algorithms分开需要考虑挺多的，因为涉及到中间要传参之类的

这里我没想好应该怎么设计。。。
scikit-learn这部分做成面向过程了（好象是这样）
觉得可能有性能和占用资源的考虑
个人觉得如果需要对数据集太大特殊处理的话  比如暂存硬盘里 分批拿出来训练的话
面向对象的编程难度可以降低

Dataset似乎对于监督学习和非监督学习应该分开设计
可以考虑不分  全都用矩阵表示
用其他变量来记录那些是x 哪些是y
"""


import numpy

'''
Class Dataset to save information and operations about data.
'''
class Dataset:
    def __init__(self):
        self.number_of_examples = None
        self.dimension_x = None
        self.dimension_y = None
        self.bool_supervised = None
        
        #self.data = numpy.array
        self.dataX = None
        self.dataY = None
        
        

    def split(self):
        pass
    
    def getSize(self):
        pass
    
    '''
    parameters:
        x_train numpy.array (number of set * number of features)
        y_train numpy.array (number of set * 1)
    '''
    def load(self,x_train,y_train):

        #check the dataset
        
        self.dataX = x_train
        self.dataY = y_train
        self.dimension_x = x_train.shape[1]
        self.dimension_y = y_train.shape[1]
        self.number_of_examples = x_train.shape[0]
        
        
    
    #def generate_batch(self):
        
    '''
    只归一化x的维度
    '''
    def normalization(self):
        pass
    
    #为SVM准备数据 并做检查
    def set_up_for_SVM(self):
        pass

class Normalization():
    def __init__(self):
          self.forwardTransform = None
          self.invertTransform = None
          self.maxVector = None
          self.minVector = None
    '''
    input_data  numpy.array
    '''
    def fit(self,input_data):
        #check input_data type
        if type(input_data) != numpy.array:
            print("ERROR:input_data is not a numpy.array!")
        

        #Use maxinum and minum to normalize

        #max is a row and each element is the max of its colonm.
        maxVector = input_data.max(axis=0)
        minVector = input_data.min(axis=0)
        
        self.maxVector = maxVector
        self.minVector = minVector
        
        #scaleVector = maxVector - minVector

        #minMatrix = numpy.tile(minVector,(input_data.shape[0],1))
        #maxMatrix = numpy.tile(maxVector,(input_data.shape[0],1))
        
        #scaleMatrix = numpy.tile(scaleVector,(input_data.shape[0],1))

        normalizedData = numpy.true_divide(input_data-minMatrix, maxMatrix-minMatrix)

        #Generate a function to calculate forwardTransform and invertTransform
        
        def fT(data):
            
            
            minMatrix = numpy.tile(self.minVector,(input_data.shape[0],1))
            maxMatrix = numpy.tile(self.maxVector,(input_data.shape[0],1))
            
            normalizedData = numpy.true_divide(input_data-minMatrix, maxMatrix-minMatrix)
            
            return normalizedData
            
            

        return fT
        
        
        