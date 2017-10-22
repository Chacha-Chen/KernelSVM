# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:21:13 2017

@author: Dyt
"""

'''
模型类  我理解的模型是指
如SVM里的超平面方程这种东西
不同的model有不同的形式与参数

class Model:
    def __init__(self):
        pass
    
    def save(self):
        pass
    
    def load(self):
        pass
    
'''


'''
如SVM中使用的超平面是一种线性模型，用这个类来表示，存储其权重和偏置
还可以定义其他方法诸如
给一组输入 计算模型的输出 即求 f(x) = w^T x + b 中f(x)的值（如果需要的话）
'''    
class LinearModel:
    def __init__(self):
        #Model.__init__(self)
        self.dimension = None
        self.weight = None
        self.bias = None
        
        pass
    
    '''
    计算模型输出的值
    这里可以采用SMO论文中的12.2.4节中提到的办法进行性能优化
    不过我觉得这是后续要考虑的事情  我们可以先实现基本功能之后再考虑性能优化
    '''
    def calulate(self):
        pass
    
    
    