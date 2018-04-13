from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sp
import random
import sys
def inputProcess():
    if len(sys.argv) == 5:
        lr = float(str(sys.argv[1]))
        epoch = int(str(sys.argv[2]))
        train = str(sys.argv[3])
        test = str(sys.argv[4])
    else:
        sys.exit('ERROR: This program takes 3 iput argument.')
    return lr,epoch,train,test

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

def sigmoid(x):
    y = float(1 / (1+np.e**(-x)))
    return y

#loss function
def crossEntropy(o,y):
    E = -y*np.log(o) - (1-y)*np.log(1-o)
    return E

def SGD(lr,weight,data,meta):
    tmpData = data.sample(frac=1).reset_index(drop=True)
    CEE = 0.0
    b = np.random.uniform(-0.01,0.01,1)
    for index, row in tmpData.iterrows():
        temp = row.tolist()
        tmpInput = [1.0] + temp[:-1]
        #the true classification
        if temp[-1] == meta['class'][1][0]:
            y = 0
        else:
            y = 1
        #calculate the output
        output = sigmoid(np.dot(tmpInput,weight))
        #calculate the error
        error = crossEntropy(output,y)
        #print(error)
        gradient = list((output-y)*a for a in tmpInput)
        deltaWeight = list(float(a*lr)for a in gradient)
        weight = list(a-b for a,b in zip(weight,deltaWeight))
        CEE += error
    correct,incorrect = innerPredict(weight,data,meta)
    #print(CEE,correct,incorrect)
    #print(weight)
    return weight,CEE,correct,incorrect

def innerPredict(weightList,data,meta):
    wrongPnum = 0
    correctPnum = 0
    for index, row in data.iterrows():
        temp = row.tolist()
        tmpInput = [1.0] + temp[:-1]
        if sigmoid(sum(list(a*b for a,b in zip(tmpInput,weightList)))) >= 0.5:
            output = 1
        else:
            output = 0
        if temp[-1] == meta['class'][1][0]:
            y = 0
        else:
            y = 1
        if output == y:
            correctPnum = correctPnum + 1
        else:
            wrongPnum = wrongPnum +1
    return correctPnum,wrongPnum

def SGDdiagonse(lr,epoch,data,meta):
    feaLen = len(meta.names())
    initialWeight = np.random.uniform(-0.01,0.01,feaLen)
    newWeight = initialWeight
    for i in range(epoch):
        newWeight,CEE,correct,incorrect = SGD(lr,newWeight,data,meta)
        print(i+1, CEE, correct, incorrect,sep='\t')
    return newWeight

def SGDpredictor(testData,trainData,metaData,lr,epoch):
    weight = SGDdiagonse(lr,epoch,trainData,metaData)
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
        activation = sigmoid(np.dot(tmpInput,weight))
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

lr,epoch,dataTrain,dataTest = inputProcess()
trainData,trainMeta = arff.loadarff(dataTrain)
trainData = pd.DataFrame(trainData)
trainData['class'] = trainData['class'].str.decode('utf-8')
testData,testMeta = arff.loadarff(dataTest)
testData = pd.DataFrame(testData)
testData['class'] = testData['class'].str.decode('utf-8')
trainData = standardize(trainData,trainMeta)
testData = standardize(testData,testMeta)
SGDpredictor(testData,trainData,trainMeta,lr,epoch)
