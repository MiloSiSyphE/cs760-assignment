from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sp
import sys
class Node:
    #Default Initializer
    def __init__(self, data = None):
        self._feature = None
        self._split = None
        self._isLeaf = False
        self._Type = None
        self._children = []
        self._parent = None
        self._label = None
        self._count = None
        self._isRoot = False
        self._isLeft = False
        #self._depth = 0

    def __repr__(self):
        return "<Node Feature:%s Split:%s Count:%s Class:%s Leaf:%s Left:%s>" % (self._feature, self._split,self._count, self._label, self._isLeaf, self._isLeft)
    def __str__(self):
        return "Feature:%s, Split:%s,Count:%s, Class:%s, Leaf:%s, Left:%s" % (self._feature, self._split,self._count, self._label, self._isLeaf,self._isLeft)

    def setCount(self,CCount):
        self._count = CCount

    def getCount(self):
        return self._count

    def isLeft(self):
        return self._isLeft

    def getFeature(self):
        return self._feature

    def getSplit(self):
        return self._split

    def getChildren(self):
        for i in self._children:
            #print(i.getFeature)
            return self._children
    def getCount(self):
        return self._count

    def getType(self):
        return self._Type

    def getParent(self):
        return self._parent

    def getLabel(self):
        return self._label

    def isLeaf(self):
        return self._isLeaf

    def isRoot(self):
        return self._isRoot

    def setRoot(self):
        self._isRoot = True

    def setChildren(self,child):
        #print(child.getFeature(),child.getSplit())

        self._children.append(child)
        #print("Children Added",child)

    def setParent(self, parent):
        self._parent = parent

    def setLabel(self, label):
        self._label = label

    def setLeaf(self):
        self._isLeaf = True

    def setLeft(self):
        self._isLeft = True

    def setSplit(self, split):
        self._split = split
        #print(self._split)

    def setFeature(self, feature):
        self._feature = feature
        #print(self._feature)

    def setType(self, Type):
        self._Type = Type

def inputProcess():
    if len(sys.argv) == 4:
        fname_train = str(sys.argv[1])
        fname_test = str(sys.argv[2])
        m = int(str(sys.argv[3]))
    else:
        sys.exit('ERROR: This program takes 3 iput argument.')
    return fname_train, fname_test, m

def isNorminal(varName):
    if metaData[varName][0] == 'nominal': return True
    if metaData[varName][0] == 'numeric': return False
    else:
        print("Please check the type of variables")

def simpleEntropy(dataFrame, x):
    countList = dataFrame[x].value_counts().tolist()
    entropy = 0
    bar = sum(countList)
    for h in countList:
        p = h/bar
        entropy = entropy + p*np.log2(p)
    entropy = 0 - entropy
    return entropy

def numrSpliter(varName,subDf):
    minVar = subDf[varName].min()
    maxVar = subDf[varName].max()
    denominator = subDf.shape[0]
    subDf = subDf.sort_values([varName])

    threCandidate = set()
    for i in range(0, denominator -1):
        itm = (subDf.iloc[i][varName] + subDf.iloc[i+1][varName]) / 2.0
        if (itm != minVar) and (itm != maxVar):
            threCandidate.add(itm)

    infoGain = []
    for threshold in threCandidate:
        subDfle = subDf[subDf[varName] <= threshold]
        subDfge = subDf[subDf[varName] >  threshold]
        numLe = subDfle[varName].count() / denominator
        numGe = subDfge[varName].count() / denominator
        conditionEntropy = numLe * simpleEntropy(subDfle, targetVar) + numGe * simpleEntropy(subDfge,targetVar)
        infoGain.append(simpleEntropy(subDf, targetVar) - conditionEntropy)
    thresholdL = dict(zip(threCandidate, infoGain))
    if not thresholdL:
        return
    threshold = max(thresholdL.items(), key = lambda x:x[1])
    candidate = [varName, threshold[0], threshold[1], metaData[varName][0]]
    return candidate

def normSpliter(varName,subDf):
    normValue = subDf[varName].value_counts().index.tolist()
    countValue = subDf[varName].value_counts().tolist()
    denominator = sum(countValue)
    ratio = [ x / denominator for x in countValue]
    partialEntropy = []
    for j in normValue:
        splitByLab = subDf[subDf[varName] == j]
        partialEntropy.append(simpleEntropy(splitByLab, targetVar))
    conditionEntropy = sum([a*b for a, b in zip(ratio, partialEntropy)])
    infoGain = simpleEntropy(subDf, targetVar) - conditionEntropy


    candidate = [varName, normValue[0], infoGain, metaData[varName][0]]
    return candidate

def determineCandidateSplits(dataFrame):
    candidateL = []
    for varName in variableL:
        subDf = dataFrame[[varName, targetVar]]
        if isNorminal(varName):
            candidateL.append(normSpliter(varName, subDf))
        else:
            candidateL.append(numrSpliter(varName, subDf))
    #print(candidateL)
    candidateL = [x for x in candidateL if x is not None]
    candidateL = sorted(candidateL, key = lambda x:x[2])
    return candidateL

def findBestSplit(dataFrame):
    candidateL = determineCandidateSplits(dataFrame)
    return max(candidateL, key = lambda x:x[2])

def Partition(candidate, data):
    partition = []
    if isNorminal(candidate[0]):
        nList = list(metaData[candidate[0]][1])
        for i in nList:
            splitData = data[data[candidate[0]] == i]
            partition.append(splitData)
    else:
        subLe = data[data[candidate[0]] <= candidate[1]]
        subGe = data[data[candidate[0]] > candidate[1]]
        partition.append(subLe)
        partition.append(subGe)
    return partition

def isPure(dataFrame):
    itm = dataFrame[targetVar].value_counts().index.tolist()
    if len(itm) == 1:
        return True
    else:
        return False

def stoppingPoint(dataFrame):
    if len(dataFrame) < m:
        return True
    elif dataFrame is None:
        return True
    elif isPure(dataFrame):
        return True
    itm = determineCandidateSplits(dataFrame)
    gain = []
    #print(itm)
    for i in itm:
        gain.append(i[2])
    gain = np.array(gain)
    if all(sp.less(gain, 0)):
        return True
    return False

def classLabelCount(dataFrame):
    count1 = 0
    count2 = 0
    for i in list(dataFrame[targetVar]):
        if i == '-':
            count1 +=1
        elif i == '+':
            count2 +=1
    labelCounts = [count1, count2]
    return labelCounts

def addNode(point, Parent,data):
    major = whatClass(data)
    nodeTemp = Node()
    nodeTemp.setLabel(major)
    countClass = classLabelCount(data)
    nodeTemp.setCount(countClass)
    nodeTemp.setFeature(point[0])
    nodeTemp.setSplit(point[1])
    nodeTemp.setType(point[3])
    nodeTemp.setParent(Parent)
    return nodeTemp

def BuiltTree(data, node):
    if stoppingPoint(data):
        node.setLeaf()
        return node
    itm = findBestSplit(data)
    if itm[3] == 'nominal':
        usedLabel = []
        allLabel = list(metaData[itm[0]][1])
        partition = Partition(itm, data)
        for i in partition:
            remainLabel = [x for x in allLabel if x not in usedLabel]
            itm[1] = remainLabel[0]
            usedLabel.append(itm[1])
            child = addNode(itm, node, i)
            node.setChildren(child)
            BuiltTree(i, child)
    else:
        partition = Partition(itm, data)
        for index, i in enumerate(partition):
            child = addNode(itm, node, i)
            if index == 0: child.setLeft()
            node.setChildren(child)
            BuiltTree(i,child)

def whatClass(data):
    classNum = classLabelCount(data)
    if classNum[0] >= classNum[1]:
        majorLabel = "-"
    else:
        majorLabel = "+"
    return majorLabel

def printTree(node, depth = 0):
    if not node.isRoot():
        theCount = node.getCount()
        theCount.sort()
        if node.getType() == 'numeric':
            if node.isLeft():
                symbol = "<="
            else:
                symbol = ">"
            Split = "%.6f" % node.getSplit()
        else:
            symbol = "="
            Split = node.getFeature()
        if node.isLeaf():
            print (depth * "|\t" + "%s %s %s [%d %d]: %s" % (node.getFeature(), symbol, Split, theCount[0],theCount[1],node.getLabel()))
            return
        else:
            print (depth * "|\t" + "%s %s %s [%d %d]" % (node.getFeature(), symbol, Split, theCount[0],theCount[1]))
        depth += 1
    for child in node.getChildren():
        printTree(child, depth)

def Predict(row, node):
    prediction = None
    if node.isLeaf():
        #print(node.getLabel())
        return node.getLabel()
    for child in node.getChildren():
        #print(child)
        #print(2)
        feature = child.getFeature()
        Type = child.getType()
        #print("Item:%s Split:%s" %(row[feature].item(),child.getSplit()))
        if Type == 'numeric':
                if child.isLeft():
                    #print(3)
                    if row[feature].item() <= child.getSplit():
                        prediction = Predict(row, child)
                else:
                    #print(4)
                    if row[feature].item() > child.getSplit():
                        #print("Worked")
                        prediction = Predict(row, child)
        else:
            #print(5)
            if row[feature].item() == child.getSplit():
                prediction = Predict(row, child)
    #print(prediction)
    return prediction

def Performance(dataTest, tree):
    #if Done:
    print ("<Predictions for the Test Set Instances>")
    correctness = 0
    dataLen = len(dataTest)
    for i in range(dataLen):
        row = dataTest.loc[[i]]
        thePrediction = Predict(row, tree)
        #print(thePrediction)
        theFact = row[targetVar].item()
        #print(theFact)
        if thePrediction == theFact:
            correctness += 1
        #if Done:
        print ("%d: Actual: %r Predicted: %r" % (i+1, theFact, thePrediction))
    print ("Number of correctly classified: %d Total number of test: %d"  % (correctness, dataLen))
    Accuracy = 1.0 * correctness / dataLen
    return Accuracy


fname_train, fname_test, m = inputProcess()

rawCtrain, metaData = arff.loadarff(fname_test)

dataTrain = pd.DataFrame(rawCtrain)

variableL = list(dataTrain)

rawCtest, metaDataT = arff.loadarff(fname_test)
dataTest = pd.DataFrame(rawCtest)

for varName in variableL:
    if isNorminal(varName): dataTrain[varName] = dataTrain[varName].str.decode("utf-8")
targetVar = variableL[-1]
variableL.remove("class")
classLabel = dataTrain[targetVar].value_counts().tolist()

for varName in list(dataTest):
    if isNorminal(varName): dataTest[varName] = dataTest[varName].str.decode("utf-8")

Root = Node()
Root.setRoot()
BuiltTree(dataTrain, Root)
printTree(Root)
Performance(dataTest, Root)
