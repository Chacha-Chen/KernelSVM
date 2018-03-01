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



import SMO

import numpy





'''
监督学习类
有基本的功能
设想是各种算法都可以从这里继承，这里的代码可以重用

class SupervisedLearning:
    
    
    初始化
    所有的监督学习均有数据集
    因此该父类拥有Dateset作为成员变量 并拥有
    后续子类无需再添加次成员
    
    将通过
    LearningMachine=SupervisedLearning()
    LearningMachine.Data.load()
    等方式进行使用
    
    def __init__(self):
        
        该类内部不需再写关于dataset处理的内容
        可以在外部直接调用Dataset的成员函数
        
        self.Data=Dataset.Dataset()
        self.Data.bool_supervised = True
        self.Data.dimension_y = 1
        pass
    
    
    
    训练核心算法
    可以内部调用也可以外部调用
    
    def train(self,x_trains,y_trains,parameters):
        pass
    
    
    训练完毕后，给定一组x，输出相应的结果
    
    def predict(self,x_predict):
        pass
    
    
    
    结果评估
    给定一组测试集，输出准确率等信息
    需要利用predict
    
    def evaluate(self,x_test,y_test):
        pass
    
    
    
    交叉检验
    包含 对数据集的分割  需要利用train 和 evaluate
    
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
        SupervisedLearning.__init__(self)
        pass
    


从基类继承 可以继承evaluate等相同功能的函数
SVM特有的（如训练算法）在这里重载
'''


class SVM():
    #def __init__(self):
    """target model"""
        
        #self.Data=Dataset.Dataset()
        #self.b = None
        #self.w = None
        
        #self.target = numpy.zeros((self.Data.number_of_examples))

    #SVM是一个包含各种参数的类 因为分类函数与alpha x y k b有关
    def __init__(self, x_train, y_train, kernel):

        self.X = x_train  # training data，m*n
        self.Y = y_train  # class label vector，1*m
        self.kernel = kernel  # kernel function: rbf OR linear OR...
        self.alphas = numpy.zeros(len(self.X))  # lagrange multiplier vector, initialized as zeros
        self.b = None  # scalar bias term
        self.gamma_ = 1


    '''
    Using cross_validation
    User can set some of the parameter.
    '''


   # def train(self, X, Y, C=[0.01,1,10,100], gamma=[0.1,0.2,0.5,1.0], kernel='rbf', tol=1e-3):
        #Cross Validation
    '''
        里面调用SMO
        
    '''
        #X   X_0 X_1.....
        # 生成10份


    def train(self,C=[0.01,1,10,100], gamma=[0.1,0.2,0.5,1.0],kernel='rbf',tol=1e-3):
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
        '''
        X_num=self.X.shape[0]
        train_index=range(X_num)
        test_size=int(X_num*0.1)+1
        for i in range(9):
            test_index=[]
            for j in range(test_size):
                randomIndex=int(numpy.random.uniform(0,len(train_index)))
                test_index.append(train_index[randomIndex])
                #del train_index[randomIndex]
            A[i]=self.X[test_index,:]
            B[i]=self.Y[test_index,:]
        A[9]=self.X.ix_[train_index]
        B[9]=self.Y.ix_[train_index]
        '''
        
        acc_best = 0
        C_best = None
        gamma_best = None
        for CVal in C:
            for gammaVal in gamma:
                avg_acc = 0
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
                    
                    SMO.GG = gammaVal
                    model= SMO.SMO_Model(X_train, Y_train, CVal,  self.kernel, gammaVal, tol=1e-3, eps=1e-3)

                    output_model=SMO.SMO(model)
                    
                    acc = SMO._evaluate(output_model,X_test,Y_test)
                    
                    avg_acc = avg_acc +acc/10
                    
                if avg_acc > acc_best:
                    acc_best = avg_acc
                    #更新C gamma
                    C_best = CVal
                    gamma_best =gammaVal
                    self.gamma = gamma_best


        #最后一遍train
        SMO.GG = gamma_best
        SVM_model = SMO.SMO(SMO.SMO_Model(X_train, Y_train, C_best, self.kernel, gamma_best, tol=1e-3, eps=1e-3))
        # 参数传递给最后生成的SVM类
        self.X = SVM_model.X
        self.Y = SVM_model.y
        self.kernel = SVM_model.kernel
        self.alphas = SVM_model.alphas
        self.b = SVM_model.b

                        # C_best = C
                        # gamma_best =gamma
						
        # (w,b) = SMO(X_train,Y_train,C_best,gamma_best,kernal,tol=1e-3)
        # self.w = w
        # self.b = b

        
        return None
		
    



    def evaluate(self,X_test,Y):
        
        Y_predict = numpy.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            Y_predict[i] = SMO.decision_function(self.alphas, self.Y, self.kernel, self.X, X_test[i], self.b,self.gamma_)
        error = Y - Y_predict
        
        mis = numpy.linalg.norm(error,0)
        
        acc = 1 - mis / Y.shape[0]
        
        return acc
    
    

    
'''
    Single training process Using SMO
    
    def _train(self,x_train,y_train,C,kernalName,eps):
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
'''
