from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sp
import random
import sys

def inputProcess():
    if len(sys.argv) == 6:
        lr = float(str(sys.argv[1]))
        h = int(str(sys.argv[2]))
        epoch = int(str(sys.argv[3]))
        train = str(sys.argv[4])
        test = str(sys.argv[5])
    else:
        sys.exit('ERROR: This program takes 3 iput argument.')
    return lr,h,epoch,train,test

def standardize(data,meta):
    for i in data:
        varName = data[i].name
        if meta[varName][0] == 'numeric':
            data[i] = (data[i] - data[i].mean()) / data[i].std()
        if meta[varName][0] == 'nominal' and varName != 'class':
            labelList = list(meta[varName][1])
            labelLen = len(labelList)
            for j in data[i]:
                a = labelList.index(j)
                j = np.zeros(labelLen)
                j[a] = 1
    return data

#object function
def sigmoid(x):
    y = float(1 / (1+np.e**(-x)))
    return y

#loss function
def crossEntropy(o,y):
    E = -y*np.log(o) - (1-y)*np.log(1-o)
    return E
#vectorlized sigmoid
vectorSigmoid = np.vectorize(sigmoid)

def NNtrain(lr,ItoHweight,HtoOweight,data,meta):
    tmpData = data.sample(frac=1).reset_index(drop=True)
    CEE = 0.0
    for index,row in tmpData.iterrows():
        #get instance
        temp = row.tolist()
        tmpInput = [1.0] + temp[:-1]
        hidden = np.dot(ItoHweight,tmpInput)
        hiddenOutput = vectorSigmoid(hidden)
        hiddenOutput = [1.0] + list(hiddenOutput)
        output = sigmoid(np.dot(hiddenOutput,HtoOweight))
        if temp[-1] == meta['class'][1][0]:
            y = 0
        else:
            y = 1
        activation = crossEntropy(output,y)
        CEE += activation
        error = y - output
        gradient = list(float(a*(1-a)) for a in hiddenOutput)
        gradientI2H = list(float(a*error*b*lr) for a,b in zip(HtoOweight,gradient))
        gradientI2H.pop(0)
        deltaH2O = list(float(lr*error*a) for a in hiddenOutput)
        deltaI2H = np.outer(gradientI2H,tmpInput)
        HtoOweight += deltaH2O
        ItoHweight += deltaI2H
    correct,incorrect = innerPredict(ItoHweight,HtoOweight,data,meta)
    #print(CEE,correct,incorrect)
    return ItoHweight,HtoOweight,CEE,correct,incorrect

def innerPredict(ItoHweight,HtoOweight,data,meta):
    wrongPum = 0
    correctPnum = 0
    for index,row in data.iterrows():
        temp = row.tolist()
        tmpInput = np.array([1.0] + temp[:-1])
        hidden = np.dot(ItoHweight,tmpInput)
        hidden = list(hidden)
        hiddenOutput = [1.0]+list(vectorSigmoid(hidden))
        if sigmoid(np.dot(hiddenOutput,HtoOweight)) >= 0.5:
            output = 1
        else:
            output =0
        if temp[-1] == meta['class'][1][0]:
            y = 0
        else:
            y = 1
        if output == y:
            correctPnum = correctPnum +1
        else:
            wrongPum = wrongPum +1
    return correctPnum,wrongPum

def NNdiagonse(lr,h,epoch,data,meta):
    feaLen = len(meta.names())
    #print(h)
    HtoOweight = np.random.uniform(-0.01,0.01,h+1)
    ItoHweight = np.random.uniform(-0.01,0.01,(h,feaLen))
    #print(ItoHweight.shape)
    for i in range(epoch):
        newItoOweight,newHtoOweight,CEE,correct,incorrect = NNtrain(lr,ItoHweight,HtoOweight,data,meta)
        print(i+1, CEE, correct, incorrect,sep='\t')
    return newItoOweight,newHtoOweight

def NNpredictor(testData,trainData,metaData,lr,h,epoch):
    ItoHweight, HtoOweight = NNdiagonse(lr,h,epoch,trainData,metaData)
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    correct = 0
    incorrect = 0
    b = np.random.uniform(-0.01,0.01,1)
    for index,row in testData.iterrows():
        temp = row.tolist()
        tmpInput = [1.0]+temp[:-1]
        #the true classification
        if temp[-1] == metaData['class'][1][0]:
            y = 0
        else:
            y = 1
        #calculate the output
        hidden = np.dot(ItoHweight,tmpInput)
        hidden = list(hidden)
        hiddenOutput = [1.0]+list(vectorSigmoid(hidden))
        output = sigmoid(np.dot(hiddenOutput,HtoOweight))
        activation = crossEntropy(output,y)
        if activation >= 0.5:
            output = 1
        else:
            output = 0
        if temp[-1] == metaData['class'][1][0]:
            y = 0
        else:
            y = 1
        if output == y:
            correct = correct + 1
        else:
            incorrect = incorrect +1
        if output ==1 and y == 1:
            tp += 1
        if output ==1 and y == 0:
            fp +=  1
        if output == 0 and y ==1:
            fn += 1
        if output == 0 and y == 1:
            tn += 1
        print(activation,output,y)
    prec = float(tp) / (tp+fp)
    rec = float(tp) / (tp+fn)
    F1 = 2 * (prec * rec) / (prec + rec)
    print (correct,incorrect)
    print(F1)
    return F1

lr,h,epoch,dataTrain,dataTest = inputProcess()
trainData,trainMeta = arff.loadarff(dataTrain)
trainData = pd.DataFrame(trainData)
trainData['class'] = trainData['class'].str.decode('utf-8')
testData,testMeta = arff.loadarff(dataTest)
testData = pd.DataFrame(testData)
testData['class'] = testData['class'].str.decode('utf-8')
trainData = standardize(trainData,trainMeta)
testData = standardize(testData,testMeta)
NNpredictor(testData,trainData,trainMeta,lr,h,epoch)
