# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:23:45 2017

@author: Dyt
"""
import Algorithms


svm = Algorithms.SVM()

svm.Data.load()

svm.Data.set_up_for_SVM()
    
svm.train()
    
svm.evaluate()
    
svm.predict()
    
