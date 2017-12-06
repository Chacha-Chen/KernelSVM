import numpy

def kernel(x1,x2,k_type):
    #x1,x2 numpy.array
	if k_type == 'rbf':
	sigma = 1.0 
	num = x1.shape[0] 
	for i in xrange(num):  
            diff = x1[i, :] - x2  
    K = exp(diff * diff.T / (-2.0 * sigma**2)) 
    '''
	if k_type == 'rbf':
        
        K = numpy.dot(x1,x2)
		'''
    return K