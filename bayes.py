'''
classifying spam email with naive Bayes
@author: wq
'''
import numpy as np

#创建词汇表
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    #取两个集合的并集
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

#分类器训练函数
#trainMatrix：训练文档矩阵，classVec：训练类别标签向量
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #计算训练的文档数目
    numWords = len(trainMatrix[0]) #计算每篇文章的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #为了应对真实情况，故这边用ones，防止为0
    p0Num = np.ones(numWords)   #词条出现数初始化为1
    p1Num = np.ones(numWords)   #词条出现数初始化为1   
    #denominator（分母）
    p0Denom = 2.0   #分母初始化为2
    p1Denom = 2.0                        
    for i in range(numTrainDocs): #遍历每个文档
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          #非侮辱类的条件概率数组
    p0Vect = np.log(p0Num/p0Denom)          #侮辱类的条件概率数组
                                            #pAb：文档属于侮辱类的概率
    return p0Vect,p1Vect,pAbusive  #返回非侮辱类，侮辱类以及文档输入侮辱类的概率

#分类函数，用其进行分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #对应元素相乘
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
# bag of set是词集模型
#词袋模型转换为向量
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  #创建一个元素全为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#文本解析，接收大字符串并将其解析为字符串列表
def textParse(bigString):    #input is big string, #output is word list
    #使用正则表达式来切割文本
    import re
    #\w表示非非单词字符,*表示0个或多个
    
    listOfTokens = re.split(r'\W*', bigString)
    #这里修改了字符串上限
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

#这种随机选择数据的一部分作为训练集，而剩余部分作为测试集的过程成为存留交叉验证（hold-out cross calidation）
#垃圾邮件过滤，导入spam与ham下的文件文本，并将其解析为词列表
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        #读取垃圾邮件
        wordList = textParse(open('email/spam/%d.txt' % i,"rb").read().decode('GBK','ignore'))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #标记垃圾邮件，1表示垃圾文件
        #读取每个非垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('email/ham/%d.txt' % i,"rb").read().decode('GBK','ignore'))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #正常邮件的标签是0
    vocabList = createVocabList(docList)# 通过正常邮件来创建词汇表
    trainingSet = list(range(50)); testSet=[] #训练集
    #随机从训练集中50条数据中选取10条作为测试集
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))  #随机选取索索引值
        testSet.append(trainingSet[randIndex])  #添加测试集的索引值
        #del删除的是变量
        del(trainingSet[randIndex])   #在训练集列表中删除添加到测试集的索引值
    trainMat=[]; trainClasses = []  #训练集矩阵，训练集标签
    for docIndex in trainingSet:  #将训练集中的每一条数据，转化为词向量
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
        #训练算法
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    #用10条测试数据，来测试分类器的准确性
    errorCount = 0
    for docIndex in testSet:        #给剩余的测试集分类
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print( "classification error",docList[docIndex]) #打印输出错判的那条数据
    print ('the accuracy rate is: ',1.0-float(errorCount)/len(testSet)) #错误率
spamTest();
