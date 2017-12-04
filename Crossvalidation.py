import os
import numpy
#加载数据的函数，用来加载本地文本数据，同时文本每行数据以”\t”隔开，最后一列为类标号
def loadDataSet(fileName):
    fr = open(fileName)
    dataMat = []; labelMat = []
    for eachline in fr:
        lineArr = []
        curLine = eachline.strip().split('\t') #remove '\n'
        for i in range(3, len(curLine)-1):
            lineArr.append(float(curLine[i])) #get all feature from inpurfile
        dataMat.append(lineArr)
        labelMat.append(int(curLine[-1])) #last one is class lable
    fr.close()
    return dataMat,labelMat
	#dataMat为纯特征矩阵，labelMat为类别标号
	
	#进行k折交叉验证
	#split_size即k值，filename为整个数据集文件，outdir则是切分的数据集的存放路径
def splitDataSet(fileName, split_size,outdir):
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    fr = open(fileName,'r')#open fileName to read
    num_line = 0
    onefile = fr.readlines()
    num_line = len(onefile)        
    arr = np.arange(num_line) #get a seq and set len=numLine
    np.random.shuffle(arr) #generate a random seq from arr
    list_all = arr.tolist()
    each_size = (num_line+1) / split_size #size of each split sets
    split_all = []; each_split = []
    count_num = 0; count_split = 0  
	#count_num 统计每次遍历的当前个数
    #count_split 统计切分次数
    for i in range(len(list_all)): #遍历整个数字序列
        each_split.append(onefile[int(list_all[i])].strip()) 
        count_num += 1
        if count_num == each_size:
            count_split += 1 
            array_ = np.array(each_split)
            np.savetxt(outdir + "/split_" + str(count_split) + '.txt',\
                        array_,fmt="%s", delimiter='\t')  #输出每一份数据
            split_all.append(each_split) #将每一份数据加入到一个list中
            each_split = []
            count_num = 0
    return split_all
	
def underSample(datafile): #只针对一个数据集的下采样
    dataMat,labelMat = loadDataSet(datafile) #加载数据
    pos_num = 0; pos_indexs = []; neg_indexs = []   
    for i in range(len(labelMat)):#统计正负样本的下标    
        if labelMat[i] == 1:
            pos_num +=1
            pos_indexs.append(i)
            continue
        neg_indexs.append(i)
    np.random.shuffle(neg_indexs)
    neg_indexs = neg_indexs[0:pos_num]
    fr = open(datafile, 'r')
    onefile = fr.readlines()
    outfile = []
    for i in range(pos_num):
        pos_line = onefile[pos_indexs[i]]    
        outfile.append(pos_line)
        neg_line= onefile[neg_indexs[i]]      
        outfile.append(neg_line)
    return outfile #输出单个数据集采样结果
	
def generateDataset(datadir,outdir): #从切分的数据集中，对其中九份抽样汇成一个,\
    #剩余一个做为测试集,将最后的结果按照训练集和测试集输出到outdir中
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    listfile = os.listdir(datadir)
    train_all = []; test_all = [];cross_now = 0
    for eachfile1 in listfile:
        train_sets = []; test_sets = []; 
        cross_now += 1 #记录当前的交叉次数
        for eachfile2 in listfile:
            if eachfile2 != eachfile1:#对其余九份欠抽样构成训练集
                one_sample = underSample(datadir + '/' + eachfile2)
                for i in range(len(one_sample)):
                    train_sets.append(one_sample[i])
        #将训练集和测试集文件单独保存起来
        with open(outdir +"/test_"+str(cross_now)+".datasets",'w') as fw_test:
            with open(datadir + '/' + eachfile1, 'r') as fr_testsets:
                for each_testline in fr_testsets:                
                    test_sets.append(each_testline) 
            for oneline_test in test_sets:
                fw_test.write(oneline_test) #输出测试集
            test_all.append(test_sets)#保存训练集
        with open(outdir+"/train_"+str(cross_now)+".datasets",'w') as fw_train:
            for oneline_train in train_sets:   
                oneline_train = oneline_train
                fw_train.write(oneline_train)#输出训练集
            train_all.append(train_sets)#保存训练集
    return train_all,test_all

	
	#计算SN和SP，用于评估交叉验证
def performance(labelArr, predictArr):#类标签为int类型
    #labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.; TN = 0.; FP = 0.; FN = 0.   
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == -1:
            FN += 1.
        if labelArr[i] == -1 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == -1 and predictArr[i] == -1:
            TN += 1.
    SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN 
    SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return SN,SP
	
	#利用smo算法对训练集和测试集进行训练
	#分类器要改为之后自己写的
def classifier(clf,train_X, train_y, test_X, test_y):#X:训练特征，y:训练标号
    # train with randomForest 
    print(" training begin...")
    clf = clf.fit(train_X,train_y)
    print(" training end.")
    #==========================================================================
    # test randomForestClassifier with testsets
    print(" test begin.")
    predict_ = clf.predict(test_X) #return type is float64
    proba = clf.predict_proba(test_X) #return type is float64
    score_ = clf.score(test_X,test_y)
    print(" test end.")
    #==========================================================================
    # Modeal Evaluation
    ACC = accuracy_score(test_y, predict_)
    SN,SP = performance(test_y, predict_)
    MCC = matthews_corrcoef(test_y, predict_)
    #AUC = roc_auc_score(test_labelMat, proba)
    #==========================================================================
    #save output 
    eval_output = []
    eval_output.append(ACC);eval_output.append(SN)  #eval_output.append(AUC)
    eval_output.append(SP);eval_output.append(MCC)
    eval_output.append(score_)
    eval_output = np.array(eval_output,dtype=float)
    np.savetxt("proba.data",proba,fmt="%f",delimiter="\t")
    np.savetxt("test_y.data",test_y,fmt="%f",delimiter="\t")
    np.savetxt("predict.data",predict_,fmt="%f",delimiter="\t") 
    np.savetxt("eval_output.data",eval_output,fmt="%f",delimiter="\t")
    print("Wrote results to output.data...EOF...")
    return ACC,SN,SP
	
	#求列表list中数值的平均值，主要是求ACC_mean,SP_mean,SN_mean，用来评估模型好坏
def mean_fun(onelist):
    count = 0
    for i in onelist:
        count += i
    return float(count/len(onelist))
	
	
def crossValidation(clf, clfname, curdir,train_all, test_all):
    os.chdir(curdir)
    #构造出纯数据型样本集
    cur_path = curdir
    ACCs = [];SNs = []; SPs =[]
    for i in range(len(train_all)):
        os.chdir(cur_path)
        train_data = train_all[i];train_X = [];train_y = []
        test_data = test_all[i];test_X = [];test_y = []
        for eachline_train in train_data:
            one_train = eachline_train.split('\t') 
            one_train_format = []
            for index in range(3,len(one_train)-1):
                one_train_format.append(float(one_train[index]))
            train_X.append(one_train_format)
            train_y.append(int(one_train[-1].strip()))
        for eachline_test in test_data:
            one_test = eachline_test.split('\t')
            one_test_format = []
            for index in range(3,len(one_test)-1):
                one_test_format.append(float(one_test[index]))
            test_X.append(one_test_format)
            test_y.append(int(one_test[-1].strip()))
        #======================================================================
        #classifier start here
        if not os.path.exists(clfname):#使用的分类器
            os.mkdir(clfname)
        out_path = clfname + "/" + clfname + "_00" + str(i)#计算结果文件夹
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        os.chdir(out_path)
        ACC, SN, SP = classifier(clf, train_X, train_y, test_X, test_y)
        ACCs.append(ACC);SNs.append(SN);SPs.append(SP)
        #======================================================================
    ACC_mean = mean_fun(ACCs)
    SN_mean = mean_fun(SNs)
    SP_mean = mean_fun(SPs)
    #==========================================================================
    #output experiment result
    os.chdir("../")
    os.system("echo `date` '" + str(clf) + "' >> log.out")
    os.system("echo ACC_mean=" + str(ACC_mean) + " >> log.out")
    os.system("echo SN_mean=" + str(SN_mean) + " >> log.out")
    os.system("echo SP_mean=" + str(SP_mean) + " >> log.out")
    return ACC_mean, SN_mean, SP_mean
	
	
	
if __name__ == '__main__':
    os.chdir("your workhome") #你的数据存放目录
    datadir = "split10_1" #切分后的文件输出目录
    splitDataSet('datasets',10,datadir)#将数据集datasets切为十个保存到datadir目录中
    #==========================================================================
    outdir = "sample_data1" #抽样的数据集存放目录
    train_all,test_all = generateDataset(datadir,outdir) #抽样后返回训练集和测试集
    print("generateDataset end and cross validation start")
    #==========================================================================
    #分类器部分
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=500) #使用随机森林分类器来训练
    clfname = "RF_1"    #==========================================================================
    curdir = "experimentdir" #工作目录
    #clf:分类器,clfname:分类器名称,curdir:当前路径,train_all:训练集,test_all:测试集
    ACC_mean, SN_mean, SP_mean = crossValidation(clf, clfname, curdir,train_all,test_all)
    print(ACC_mean,SN_mean,SP_mean)  #将ACC均值，SP均值，SN均值都输出到控制台