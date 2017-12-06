def train(self,X,Y,C=[0.01,1,10,100], gamma=[0.1,0.2,0.5,1.0],kernal='rbf',tol=1e-3):
        #Cross Validation
        '''
        里面调用SMO
        
        '''
		#X   X_0 X_1.....
        # 生成10份
        X_num=X.shape[0]
		train_index=range(X_num)
		test_size=int(X_num*0.1)+1
		for i in range(9):
		    test_index=[]
		    for j in range(test_size)
			    randomIndex=int(np.random.uniform(0,len(train_index)))
                test_index.append(train_index[randomIndex])
                del train_index[randomIndex]
		    A[i]=X.ix[test_index]
			B[i]=Y.ix[test_index]
		A[9]=X.ix[train_index]
        B[9]=Y.ix[train_index]		
				
        acc_best = 0
        C_best = None
        gamma_best = None
        for CVal in C:
            for gammaVal in gamma:
                for i in range(10):
                    X_test = A[i]
                    Y_test = B[i]
                    
                    X_train = np.concatenate([A[(i+1)%10],A[(i+2)%10],A[(i+3)%10],A[(i+4)%10],A[(i+5)%10],A[(i+6)%10],A[(i+7)%10],A[(i+8)%10],A[(i+9)%10]], axis=0)
                    Y_train = np.concatenate([B[(i+1)%10],B[(i+2)%10],B[(i+3)%10],B[(i+4)%10],B[(i+5)%10],B[(i+6)%10],B[(i+7)%10],B[(i+8)%10],B[(i+9)%10]], axis=0)
                    
                    (w,b) = SMO.SMO(X_train,Y_train,CVal,gammaVal,kernal,tol=1e-3)
                    
                    acc = _evaulate(w,b,X_test,Y_test)
                    
                    if acc > acc_best:
                        acc_best = acc
                        #更新C gamma
                        C_best = C
						gamma_best =gamma
						
        (w,b) = SMO(X_train,Y_train,C_best,gamma_best,kernal,tol=1e-3)
        self.w = w
        self.b = b
        
        return None
		
