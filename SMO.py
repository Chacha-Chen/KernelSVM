# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:10:40 2017
SMO算法模块
Sequential Minimal Optimization (SMO)

基类：监督学习机

子类：支持向量机

@author: Chacha


objective_function              优化目标函数
decision_function               决策函数
SMO_Model                       保存SMO算法需要用到的所有参数信息
take_step & examine_example     进行SMO运算所需要的工具函数
SMO(model)                      对model进行SMO算法的优化
_evaulate(output_model,X_test,Y_test)
                                计算准确率
"""

import numpy as np


# Objective function to optimize 优化目标函数
# 总感觉有问题 需要处理一下
def objective_function(alphas, target, kernel, X_train):
    """Returns the SVM objective function"""
    result = 0
    for i in range(X_train.shape[0]):      #m个数据
        for j in range(X_train.shape[0]):
        result -= 0.5 * target[i] * target[j] * kernel(X_train[i], X_train[j]) * alphas[i] * alphas[j]

    result += np.sum(alphas)
    return result



# Decision function 分类函数
def decision_function(alphas, target, kernel, X_train, X_test, b):
    """input `x_test` return y."""

    result = np.dot((alphas * target) , kernel(X_train, X_test)) - b
    return result


class SMO_Model:
    #initialization
    def __init__(self, x_train, y_train, C, kernel, gammaVal, tol, eps):
        self.X = x_train                      # training data，m*n
        self.y = y_train                      # class label vector，1*m
        self.C = C                            # punishment factor
        self.kernel = kernel                  # kernel function: rbf OR linear OR...
        self.alphas = np.zeros(len(self.X))   # lagrange multiplier vector, initialized as zeros
        self.b = None                         # scalar bias term
        self.errors =np.zeros(len(self.y))    # error cache, initialized as zeros
        #self._obj = []                       # record of objective function value
        self.m = len(self.X)                  # store size of training set
        self.gammaVal = gammaVal              # kernel计算参数
        self.tol = tol                        # error tolerance
        self.eps = eps                        # alpha tolerance

def take_step(i1, i2, model):
    # Skip if chosen alphas are the same
    if i1 == i2:
        return 0, model

    alph1 = model.alphas[i1]
    alph2 = model.alphas[i2]
    y1 = model.y[i1]
    y2 = model.y[i2]
    E1 = model.errors[i1]
    E2 = model.errors[i2]
    s = y1 * y2

    # Compute L & H, the bounds on new possible alpha values
    if (y1 != y2):
        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif (y1 == y2):
        L = max(0, alph1 + alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if (L == H):
        return 0, model

    # Compute kernel & 2nd derivative eta
    k11 = model.kernel(model.X[i1], model.X[i1], model.gammaVal)
    k12 = model.kernel(model.X[i1], model.X[i2], model.gammaVal)
    k22 = model.kernel(model.X[i2], model.X[i2], model.gammaVal)
    eta = 2 * k12 - k11 - k22

    # Compute new alpha 2 (a2) if eta is negative
    if (eta < 0):
        a2 = alph2 - y2 * (E1 - E2) / eta
        # Clip a2 based on bounds L & H
        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H

    # If eta is non-negative, move new a2 to bound with greater objective function value
    else:
        alphas_adj = model.alphas.copy()
        alphas_adj[i2] = L

        # objective function output with a2 = L
        Lobj = objective_function(alphas_adj, model.y, model.kernel, model.X)

        alphas_adj[i2] = H
        # objective function output with a2 = H
        Hobj = objective_function(alphas_adj, model.y, model.kernel, model.X)
        if Lobj > (Hobj + model.eps):
            a2 = L
        elif Lobj < (Hobj - model.eps):
            a2 = H
        else:
            a2 = alph2

    # Push a2 to 0 or C if very close
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C

    # If examples can't be optimized within epsilon (eps), skip this pair
    if (np.abs(a2 - alph2) < model.eps * (a2 + alph2 + model.eps)):
        return 0, model

    # Calculate new alpha 1 (a1)
    a1 = alph1 + s * (alph2 - a2)

    # Update threshold b to reflect newly calculated alphas
    # Calculate both possible thresholds
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b

    # Set new threshold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < model.C:
        b_new = b1
    elif 0 < a2 and a2 < model.C:
        b_new = b2
    # Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5

    # Update model object with new alphas & threshold
    model.alphas[i1] = a1
    model.alphas[i2] = a2

    # Update error cache
    # Error cache for optimized alphas is set to 0 if they're unbound
    for index, alph in zip([i1, i2], [a1, a2]):
        if 0.0 < alph < model.C:
            model.errors[index] = 0.0

    # Set non-optimized errors based on equation 12.11 in Platt's book
    non_opt = [n for n in range(model.m) if (n != i1 and n != i2)]
    model.errors[non_opt] = model.errors[non_opt] + \
                            y1 * (a1 - alph1) * model.kernel(model.X[i1], model.X[non_opt], model.gammaVal) + \
                            y2 * (a2 - alph2) * model.kernel(model.X[i2], model.X[non_opt], model.gammaVal) + model.b - b_new

    # Update model threshold
    model.b = b_new

    return 1, model


def examine_example(i2, model):
    y2 = model.y[i2]
    alph2 = model.alphas[i2]
    E2 = model.errors[i2]
    r2 = E2 * y2

    # Proceed if error is within specified tolerance (tol)
    if ((r2 < -model.tol and alph2 < model.C) or (r2 > model.tol and alph2 > 0)):

        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            # Use 2nd choice heuristic is choose max difference in error
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # Loop through non-zero and non-C alphas, starting at a random point
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # loop through all alphas, starting at a random point
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

    return 0, model

#TO_CALCULATE SMO
#OUTPUT: SMO_Model

def SMO(model):
    numChanged = 0
    examineAll = 1

    while (numChanged > 0) or (examineAll):
        numChanged = 0
        if examineAll:
            # loop over all training examples
            for i in range(model.alphas.shape[0]):
                examine_result, model = SMO_Model.examine_example(i, model)
                numChanged += examine_result
                #if examine_result:
                    #obj_result = SMO_Model.objective_function(model.alphas, model.y, model.kernel, model.X)
                    #model._obj.append(obj_result)
        else:
            # loop over examples where alphas are not already at their limits
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = SMO_Model.examine_example(i, model)
                numChanged += examine_result
                #if examine_result:
                    #obj_result = SMO_Model.objective_function(model.alphas, model.y, model.kernel, model.X)
                    #model._obj.append(obj_result)
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1

    return model


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
'''

#acc = _evaulate(output_model,X_test,Y_test)
def _evaluate(output_model,X_test,Y_test):
    Y_predict = decision_function(output_model.alphas, output_model.Y_train, output_model.kernel, output_model.X_train, X_test, output_model.b)
    error = Y_test - Y_predict
        
    mis = np.linalg.norm(error,0)
        
    acc = 1 - mis / Y_test.shape[0]
        
    return acc
    