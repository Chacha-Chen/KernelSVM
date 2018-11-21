# KernelSVM

A python toolbox for support vector machine on both indefinite kernel learning and traditional methods(semi-definite)
-------------

@[Leyu Yao](https://github.com/dynasting), @[Zhebing Huang](https://github.com/793159320), @[Chacha Chen](https://github.com/chacha-chen)

### Usage

First define a kernel dictionary must be defined before calling algorithms. A kernel type should be specified with sufficient kernel parameters.
```
kernel_dict = {'type':'[kernel_type]', 'c' : arg1, 'd':arg2}
```
\[kernel_type\] options:
- RBF: gaussian kernel
- LINEAR: linear kernel
- POLY: polynomial kernel
- TAHN: tahn kernel
- TL1: TL1 kernel

SVM is implemented in 'Algorithm.py', call SVM simple by
```
import Algorithms

svm = Algorithms.SVM(X_train, Y_train, kernel_dict)
svm.train(C=[0.1,1,10])
```
Similarly, KVM, LSSVM, and kPCA could be called via corresponding module.

### Environment and Dependencies:
![](https://img.shields.io/badge/python-3.6-brightgreen.svg)

[![](https://img.shields.io/badge/anaconda3-4.4.0-brightgreen.svg)](https://www.anaconda.com/download/)

<!--![](https://img.shields.io/badge/platform-Windows10-blue.svg)-->



### Projects Framework  
- Algorithms: Core module for learning  

- Dataset: Module for dataset processing  

- Kernel: Module for a variety of kernels  

- Model: Module for a variety of models, e.g.  linear model  

- SMO: Module of SMO algorithms


<!--### 项目进展    
SVM SMO基础学习  
工具箱架构设计    
传统SMO算法实现   
核函数计算  
####  SMO算法优化  
####  非正定核的SVM
####  非正定核的LSSVM和PCA-->


### Reference

[X. Huang, A. Maier, J. Hornegger, J.A.K. Suykens: Indefinite Kernels in Least Squares Support Vector Machine and Principal Component Analysis, Applied and Computational Harmonic Analysis, 43(1): 162-172, 2017](https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2016/Huang16-IKI.pdf)

[Schleif, Frank-Michael, and Peter Tino. "Indefinite proximity learning: A review." Neural computation 27.10 (2015): 2039-2096.](https://www.techfak.uni-bielefeld.de/~fschleif/ijcnn_2015/NECO-02-015-2298-Source.pdf)

[Boser, Bernhard E., Isabelle M. Guyon, and Vladimir N. Vapnik. "A training algorithm for optimal margin classifiers." Proceedings of the fifth annual workshop on Computational learning theory. ACM, 1992.
](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.21.3818&rep=rep1&type=pdf)
 
[Platt, John. "Sequential minimal optimization: A fast algorithm for training support vector machines." (1998)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)

[Luss, R., & d'Aspremont, A. (2008). Support vector machine classification with indefinite kernels. In Advances in Neural Information Processing Systems (pp. 953-960).](http://papers.nips.cc/paper/3339-support-vector-machine-classification-with-indefinite-kernels.pdf)
