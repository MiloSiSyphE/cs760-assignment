#====load libraries====
from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sp
import sys
from decimal import *
import itertools
from itertools import permutations
import operator


def inputProcess():
    fname_train = str(sys.argv[1])
    fname_test = str(sys.argv[2])
    patern = str(sys.argv[3])
    return fname_train,fname_test,patern

fname_train,fname_test,patern = inputProcess()
dataTrain, metaTrain = arff.loadarff(fname_train)
dataTest, metaTest = arff.loadarff(fname_test)
dataTrain = pd.DataFrame(dataTrain)
dataTest = pd.DataFrame(dataTest)


def trainCMI(dataTrain):
    CMIMatrix = pd.DataFrame(columns = varTest,index = varTest)
    for i in combList:
        X1 = i[0]
        X2 = i[1]
        itm = CMI(X1,X2)
        CMIMatrix[X1][X2] = itm.item()
    for i in varTest:
        CMIMatrix[i][i] = -1
    #CMIDict = [(k, CMIDict[k]) for k in sorted(CMIDict, key = CMIDict.get, reverse = True)]
    return CMIMatrix

def CMI(X1,X2,Y="class",dataTrain = dataTrain,metaTrain = metaTrain):
    X1len = len(metaTrain[X1][1])
    X2len = len(metaTrain[X2][1])
    X1List = list(metaTrain[X1][1])
    X2List = list(metaTrain[X2][1])
    YList = list(metaTrain[Y][1])
    toCompute = [X1List,X2List,YList]
    paraList = list(itertools.product(*toCompute))
    itmList = []
    for i in paraList:
        itm = PXXY(X1,X2,i,X1len,X2len)*np.log2(np.divide(PXX__Y(X1,X2,i,X1len,X2len),PX__Y(X1,X2,i,X1len,X2len),dtype=float))
        itmList.append(itm)
        #print(itm)
    CMI = sum(itmList)
    return CMI


def PXXY(X1,X2,values,X1len,X2len,Y='class'):
    XXYcount = len(dataTrain.loc[(dataTrain[X1] == values[0]) & (dataTrain[X2] == values[1]) & (dataTrain[Y] == values[2])])
    XXYcount = XXYcount + 1
    Ycount = len(dataTrain) + X1len * X2len * 2
    PXXY = np.divide(XXYcount, Ycount, dtype=float)
    return PXXY
def PXX__Y(X1,X2,values,X1len,X2len,Y='class'):
    XXYcount = len(dataTrain.loc[(dataTrain[X1] == values[0]) & (dataTrain[X2] == values[1]) & (dataTrain[Y] == values[2])])
    XXYcount = XXYcount + 1
    Ycount = len(dataTrain.loc[(dataTrain[Y] == values[2])])
    Ycount = Ycount + X1len * X2len
    PXX__Y = np.divide(XXYcount,Ycount,dtype = float)
    return PXX__Y
def PX__Y(X1,X2,values,X1len,X2len,Y='class'):
    X1YCount = len(dataTrain.loc[(dataTrain[X1] == values[0]) & (dataTrain[Y] == values[2])]) + 1
    X2YCount = len(dataTrain.loc[(dataTrain[X2] == values[1]) & (dataTrain[Y] == values[2])]) + 1
    Ycount = len(dataTrain.loc[dataTrain[Y] == values[2]])
    Ycount1 = Ycount + X1len
    Ycount2 = Ycount + X2len
    PX1__Y = np.divide(X1YCount, Ycount1, dtype=float)
    PX2__Y = np.divide(X2YCount, Ycount2, dtype=float)
    PX__Y = PX1__Y * PX2__Y
    return PX__Y

def findMST(trainMartix, varTest):
    vSet = varTest[:]
    vNew = ['lymphatics']
    vRoot = {}
    eNew = []
    while vSet != []:
        base = 0
        outVec = 0
        inVec = 0
        for u in vNew:
            for v in vSet:
                if trainMatrix[u][v] > base:
                    base = trainMatrix[u][v]
                    outVec = u
                    inVec = v
                if trainMatrix[u][v] == base:
                    if (u <= outVec) and (v <= inVec):
                        outVec = u
                        inVec = v
        if inVec in vSet: vSet.remove(inVec)
            #vSet.remove(inVec)
        vNew.append(inVec)
        vRoot[inVec] = outVec
        item = [inVec,outVec]
        eNew.append(item)
    return eNew

def trainCMI(dataTrain):
    CMIMatrix = pd.DataFrame(columns = varTest,index = varTest)
    for i in combList:
        X1 = i[0]
        X2 = i[1]
        itm = CMI(X1,X2)
        CMIMatrix[X1][X2] = itm.item()
    for i in varTest:
        CMIMatrix[i][i] = -1
    #CMIDict = [(k, CMIDict[k]) for k in sorted(CMIDict, key = CMIDict.get, reverse = True)]
    return CMIMatrix

def CMI(X1,X2,Y="class",dataTrain = dataTrain,metaTrain = metaTrain):
    X1len = len(metaTrain[X1][1])
    X2len = len(metaTrain[X2][1])
    X1List = list(metaTrain[X1][1])
    X2List = list(metaTrain[X2][1])
    YList = list(metaTrain[Y][1])
    toCompute = [X1List,X2List,YList]
    paraList = list(itertools.product(*toCompute))
    itmList = []
    for i in paraList:
        itm = PXXY(X1,X2,i,X1len,X2len)*np.log2(np.divide(PXX__Y(X1,X2,i,X1len,X2len),PX__Y(X1,X2,i,X1len,X2len),dtype=float))
        itmList.append(itm)
        #print(itm)
    CMI = sum(itmList)
    return CMI


def PXXY(X1,X2,values,X1len,X2len,Y='class'):
    XXYcount = len(dataTrain.loc[(dataTrain[X1] == values[0]) & (dataTrain[X2] == values[1]) & (dataTrain[Y] == values[2])])
    XXYcount = XXYcount + 1
    Ycount = len(dataTrain) + X1len * X2len * 2
    PXXY = np.divide(XXYcount, Ycount, dtype=float)
    return PXXY
def PXX__Y(X1,X2,values,X1len,X2len,Y='class'):
    XXYcount = len(dataTrain.loc[(dataTrain[X1] == values[0]) & (dataTrain[X2] == values[1]) & (dataTrain[Y] == values[2])])
    XXYcount = XXYcount + 1
    Ycount = len(dataTrain.loc[(dataTrain[Y] == values[2])])
    Ycount = Ycount + X1len * X2len
    PXX__Y = np.divide(XXYcount,Ycount,dtype = float)
    return PXX__Y
def PX__Y(X1,X2,values,X1len,X2len,Y='class'):
    X1YCount = len(dataTrain.loc[(dataTrain[X1] == values[0]) & (dataTrain[Y] == values[2])]) + 1
    X2YCount = len(dataTrain.loc[(dataTrain[X2] == values[1]) & (dataTrain[Y] == values[2])]) + 1
    Ycount = len(dataTrain.loc[dataTrain[Y] == values[2]])
    Ycount1 = Ycount + X1len
    Ycount2 = Ycount + X2len
    PX1__Y = np.divide(X1YCount, Ycount1, dtype=float)
    PX2__Y = np.divide(X2YCount, Ycount2, dtype=float)
    PX__Y = PX1__Y * PX2__Y
    return PX__Y


def PX_Y(X,Xvalue,dataTrain,Y='class'):
    Xcount = len(metaTest[X][1])
    PXYL = list()
    for i in classLabel:
        XYcount = len(dataTrain.loc[(dataTrain[targetValue] == i ) & (dataTrain[X] == Xvalue)].index)
        XYcount = np.add(XYcount,1.0)
        Ycount = len(dataTrain.loc[dataTrain[targetValue] == i])
        PXY = np.divide(XYcount,Ycount + Xcount,dtype= float )
        #print(PXY)
        PXYL.append(PXY)
    np.asarray(PXYL,dtype=float)
    return PXYL

def PX_XY(X,Root,Xvlaue,RootValue,dataTrain,Y='class'):
    Xcount = len(metaTest[X][1])
    PX_XY = list()
    for i in classLabel:
        XXYcount = len(dataTrain.loc[(dataTrain[X] == Xvlaue) & (dataTrain[Root] == RootValue) & (dataTrain[Y] == i)].index) + 1
        XYcount = len(dataTrain.loc[(dataTrain[Root] == RootValue) & (dataTrain[Y] == i)]) + Xcount
        itm = np.divide(XXYcount, XYcount, dtype = float)
        PX_XY.append(itm)
    np.asarray(PX_XY, dtype = float)
    return PX_XY

def PY():
    Allcount = len(dataTrain)
    PYL = list()
    for i in classLabel:
        Ycount = len(dataTrain.loc[(dataTrain[targetValue]==i)])
        PY = np.divide(Ycount + 1 , Allcount + 2, dtype = float)
        PYL.append(PY)
    PYL = np.asarray(PYL)
    return PYL

def naiveBayes(dataTest):
    predictionL = list()
    estimateProL = np.zeros(shape = len(dataTest))
    PYL = PY()
    PY0 = PYL[0]
    PY1 = PYL[1]
    for index,row in dataTest.iterrows():
        prob = np.array([1.0,1.0], dtype = float)
        for i in varTest:
            value = row[i]
            #print(row['class'])
            #print(value)
            tempProb = PX_Y(i, value, dataTrain)
            prob = np.multiply(prob, tempProb)
            #for i in prob:
            #    print(i)
        PB0 = np.multiply(PY0,prob[0])
        PB1 = np.multiply(PY1,prob[1])
        if PB0 <= PB1:
            predictionL.append(classLabel[1])
            Temp = np.divide(PB1,PB0 + PB1, dtype = float)
            estimateProL[index] = Temp
        else:
            predictionL.append(classLabel[0])
            Temp = np.divide(PB0, PB0 +PB1,dtype = float)
            estimateProL[index] = Temp
    estimateProL = estimateProL.tolist()
    return predictionL, estimateProL

def TAN(dataTest,MST):
    predictionL = list()
    estimateProL = np.zeros(shape = len(dataTest))
    PYL = PY()
    PY0 = PYL[0]
    PY1 = PYL[1]
    for index, row in dataTest.iterrows():
        prob = np.array([1.0,1.0], dtype = np.float64)
        for i in varTest:
            value = row[i]
            if i == 'lymphatics':
                tempProb = PX_Y(i,value,dataTrain)
            else:
                for subList in MST:
                    if subList[0] == i:
                        Root = subList[1]
                rootValue = row[Root]
                tempProb = PX_XY(i,Root,value,rootValue,dataTrain)
            prob = np.multiply(prob, tempProb)
        PB0 = np.multiply(PY0,prob[0])
        PB1 = np.multiply(PY1,prob[1])
        if PB0 <= PB1:
            predictionL.append(classLabel[1])
            Temp = np.divide(PB1,PB0+PB1,dtype = np.float64)
            estimateProL[index] = Temp
        else:
            predictionL.append(classLabel[0])
            Temp = np.divide(PB0, PB0 +PB1,dtype = np.float64)
            estimateProL[index] = Temp
    estimateProL = estimateProL.tolist()
    return predictionL, estimateProL

def tanPrint(result):
    counter = 0
    for i in varTest:
        if i == varTest[0]:
            print(i,varTrain[-1])
        else:
            for subList in MST:
                if subList[0] == i:
                    Root = subList[1]
            print(i,Root,varTrain[-1])
    print()
    for i in range(len(dataTest)):
        if result[0][i] == dataTest[targetValue][i]:
            counter = counter +1
        print(result[0][i],dataTest[targetValue][i],"%.12f" % round(result[1][i],12))
    print()
    print(counter)

def nbPrint(result):
    counter = 0
    for i in varTest:
        print(i,varTrain[-1])
    print()
    for i in range(len(dataTest)):
        if result[0][i] == dataTest[targetValue][i]:
            counter = counter +1
        print(result[0][i], dataTest[targetValue][i], "%.12f" % round(result[1][i],12))
    print()
    print(counter)

#===============================================


varTrain = list(dataTrain)
varTest = list(dataTest)
targetValue = 'class'
for i in varTrain:
    dataTrain[i] = dataTrain[i].str.decode("utf-8")
for i in varTest:
    dataTest[i] = dataTest[i].str.decode("utf-8")
varTest.remove(targetValue)
classLabel = dataTrain['class'].value_counts().index.tolist()
a = permutations(varTest,2)
combList = list(a)
trainMatrix = trainCMI(dataTrain)
MST = findMST(trainMatrix, varTest)

if patern == 't':
    result = TAN(dataTest,MST)
    tanPrint(result)
if  patern == 'n':
    result = naiveBayes(dataTest)
    nbPrint(result)
