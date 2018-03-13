# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:10:40 2017
整合后的SMO算法模块
Integrated Sequential Minimal Optimization (SMO)

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
GG = 0.1


def objective_function(alphas, target, kernel, X_train):
    """Returns the SVM objective function"""
    result = 0

    result -= 0.5 * np.sum(np.multiply(np.multiply(target.T * target,kernel.kernelMat), alphas.T * alphas))
    result += np.sum(alphas)
    return result


# Decision function 分类函数
def decision_function(alphas, target, kernel, X_train, X_test, b):
    """input `x_test` return y."""
    kernel.expand(np.matrix(X_test))
    A = np.multiply(alphas,target)
    result = -b + np.dot(kernel.testMat,A)
    if result >= 0:
        return 1
    else:
        return -1

                         
def _decision_function(alphas, target, kernel, X_train, X_test, b):
    """input `x_test` return y."""

    kernel.expand(np.matrix(X_test))
    A = np.multiply(alphas,target)
    result = -b + np.dot(kernel.testMat,A)
    
    return result            


class SMO_Model:
    #initialization
    def __init__(self, x_train, y_train, C, kernel, tol, eps):
        self.X = x_train                      # training data，m*n
        self.y = y_train                      # class label vector，1*m
        self.C = C                            # punishment factor
        self.kernel = kernel                  # kernel class
        self.alphas = np.zeros(len(self.X))   # lagrange multiplier vector, initialized as zeros
        self.b = 0                            # scalar bias term
        self.errors =np.zeros(len(self.y))    # error cache, initialized as zeros
        self.m = len(self.X)                  # store size of training set
        self.tol = tol                        # error tolerance
        self.eps = eps
        self.errors =_decision_function(self.alphas, self.y, self.kernel, self.X, self.X, self.b) - self.y
        self.errors = np.array(self.errors)
        self.errors = self.errors.reshape(self.m,)

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
    k11 = model.kernel.call(i1,i1)
    k12 = model.kernel.call(i1,i2)
    k22 = model.kernel.call(i2,i2)
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
    for i in range(len(non_opt)):
        model.errors[non_opt[i]] = model.errors[non_opt[i]] + \
                            y1 * (a1 - alph1) * model.kernel.call(i1,non_opt[i]) + \
                            y2 * (a2 - alph2) * model.kernel.call(i2,non_opt[i]) + model.b - b_new

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
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
        else:
            # loop over examples where alphas are not already at their limits
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1

    return model



def _evaluate(output_model,X_test,Y_test):
    Y_predict = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        Y_predict[i] = decision_function(output_model.alphas, output_model.y, output_model.kernel, output_model.X, X_test[i], output_model.b)
       
    error = Y_test - Y_predict
        
    mis = np.linalg.norm(error,0)
        
    acc = 1 - mis / Y_test.shape[0]
        
    return acc


    


    