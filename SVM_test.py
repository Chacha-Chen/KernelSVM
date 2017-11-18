# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:23:45 2017

@author: Dyt
"""
import Algorithms


svm = Algorithms.SVM()

svm.Data.load()

svm.Data.set_up_for_SVM()
    
svm.clf(c=1,kernal='')

svm.cross_validation(clf,)

svm.clf(c=0.1,kernal='')

svm.cross_validation( )

svm.train(c=0.01,kernal='')

svm.cross_validation()


svm.evaluate()

    
svm.predict()
    
