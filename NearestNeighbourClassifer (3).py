import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
from random import randint
import math
np.set_printoptions(threshold=np.nan)

trainingImage  = []
trainingLabel = []
trainingImageGroups = []
trainingLabelGroups = []
fold = None
knn = None
bestModel = None
bestModelIndex = None
predictLabel = None
trainingAccuracy = None
trainingImage = None
trainingLabel = None

def NearestNeighbour(trainImage, trainLabel, computeGroup = False):
    global trainingImage
    global trainingLabel
    global trainingImageGroups
    global trainingLabelGroups
    global fold
    global knn
    global bestModel
    global bestModelIndex
    global predictLabel
    global trainingAccuracy
    global trainingImage
    global trainingLabel
    global fold

    trainingImage  = []
    trainingLabel = []

    if(computeGroup):
        trainingImageGroups = []
        trainingLabelGroups = []

    fold = 5
    knn = [1, 3, 5, 7, 9]
    bestModel = None
    bestModelIndex = None

    predictLabel = None
    trainingAccuracy = None
    
    trainingImage = trainImage
    trainingLabel = trainLabel
    
    if(computeGroup):
        totalCount = len(trainLabel)
    
        itemPerGroup = totalCount/float(fold)
        k = 0
        for i in range(fold):
            tempImageGroup = []
            tempLabelGroup = []
        
            for j in range(int(itemPerGroup)):
                tempImageGroup.append(trainImage[k])
                tempLabelGroup.append(trainLabel[k])
                k = k+1
                trainingImageGroups.append(tempImageGroup)
                trainingLabelGroups.append(tempLabelGroup)
    

    #identify best model
def best_model(predtictionMatrix):
    global knn
    global bestModelIndex
    global bestModel
    global trainingAccuracy

    knnCount = []
    maxKnn = 0
    maxIndx = 0
    for i in range(len(knn)-1):
        tempCount = 0
        for j in range(len(predtictionMatrix)):
            if(predtictionMatrix[j][i] == predtictionMatrix[j][-1]):
                tempCount = tempCount+1
        knnCount.append(tempCount)
        
    for i in range(len(knnCount)):
        if knnCount[i] >= maxKnn:
            maxKnn = knnCount[i]
            maxIndx = i
        
    bestModelIndex = maxIndx
    bestModel = knn[maxIndx]
    trainingAccuracy = float(maxKnn*100)/len(predtictionMatrix)
    
    return trainingAccuracy
        
    #find max occurance of element in sortes list
def max_occurance(sortedLst):
    
    tempElem = sortedLst[0][1]
    maxElem = sortedLst[0][1]
    
    maxCount = 1
    tempCount = 1
    
    for val in sortedLst[1:]:
        if(val[1]==tempElem):
            tempCount=tempCount+1
        else:
            if tempCount>=maxCount:
                maxCount = tempCount
                maxElem = tempElem
        
            tempElem = val[1]
            tempCount = 1
            
    if tempCount>=maxCount:
        maxCount = tempCount
        maxElem = tempElem
    
    return maxElem
     
    #predict image on basis of best model
def predictTestDigit(testImage):
    global predictDigit
    global bestModelIndex
    classifications = predictDigit(testImage)
    return classifications[bestModelIndex]
    
    #predic image for all models
def predictDigit(testImage):
    global trainingLabel
    global trainingImage    
    global knn
    global max_occurance 
    nearestNeighbourClassification = []
    eucledianMatrix = [[0 for x in range(2)] for y in range(len(trainingLabel))]
    k = 0
    
    for i in range(len(trainingLabel)):
        tempDistance = 0
    
        for j in range(28*28):
        	tempDistance = tempDistance+((int(trainingImage[i][j]) - int(testImage[j]))**2)
    
        eucledianMatrix[k][0] = math.sqrt(tempDistance)
        eucledianMatrix[k][1] = trainingLabel[i]
    
        k = k+1
    
    eucledianMatrix.sort()
    
    for k in knn:
        nearestNeighbourClassification.append(max_occurance(eucledianMatrix[:k]))
    
    return nearestNeighbourClassification
    
#perform cross validation for each group
def performCrossValidation(trainingImage, trainingLabel):
    #initialise class data
    global trainingImageGroups
    global trainingLabelGroups
    global fold
    NearestNeighbour(trainingImage, trainingLabel, True)
    predictedKnnMatrix = []
    for k in range(fold):
        j = 0
        trainingGroupImage = []
        trainingGroupLabel = []
        testGroupImage = []
        testGroupLabel = []
        for i in range(fold):
            if(j!=i):
                trainingGroupImage = trainingGroupImage+trainingImageGroups[i]
                trainingGroupLabel = trainingGroupLabel+trainingLabelGroups[i]
            else:
                testGroupImage = testGroupImage+trainingImageGroups[i]
                testGroupLabel = testGroupLabel+trainingLabelGroups[i]
        j = j+1
        #test for one fold
        trainingGroupImage = np.array(trainingGroupImage)
        trainingGroupLabel = np.array(trainingGroupLabel)
        testGroupImage = np.array(testGroupImage)
        testGroupLabel = np.array(testGroupLabel)
        
        NearestNeighbour(trainingGroupImage, trainingGroupLabel)
    
        for i in range(len(testGroupLabel)):
            tempKnn = predictDigit(testGroupImage[i])
            tempKnn.append(testGroupLabel[i])
            predictedKnnMatrix.append(tempKnn)
        
    return predictedKnnMatrix

 
#fetch all images from file      
def fetchImage(fileName):
    training_images_file = open(fileName,'rb') 
    training_images = training_images_file.read()
    training_images_file.close()

    training_images = bytearray(training_images)
    training_images = training_images[16:]

    image_array = np.array(training_images)
    image_array = np.reshape(image_array, (-1, 28*28))
    
    return image_array


#fetch all labels from file  
def fetchLabel(fileName):
    training_label_file = open(fileName,'rb') 
    training_label = training_label_file.read()
    training_label_file.close()

    training_label = bytearray(training_label)
    training_label = training_label[8:]

    label_array = np.array(training_label)
    
    return label_array


#fetch required number of class
def fetchClass(trainingLabel, requiredCount, classVal):
    listData = []

    count = 0
    while (count < requiredCount):
        indx = randint(0, 60000-1)
        if trainingLabel[indx] == classVal:
            listData.append(indx)
            count = count+1

    return listData

def extractTrainingData(trainingImage, trainingLabel, indexList):
    totalImages = len(indexList)
    imageList = [[-1 for x in range(28*28)] for y in range(totalImages)] 
    labelList = [-1]*totalImages
    
    for indx in indexList:
        insertIndex = 0
        while labelList[insertIndex] != -1:
            insertIndex = randint(0, totalImages-1)
        
        imageList[insertIndex] = trainingImage[indx]
        labelList[insertIndex] = trainingLabel[indx]

    imageList = np.array(imageList)
    labelList = np.array(labelList)
    return imageList, labelList


def accuracy(predtictionMatrix):
    tempCount = 0

    for j in range(len(predtictionMatrix)):
        
        if(predtictionMatrix[j][0] == predtictionMatrix[j][1]):
            tempCount = tempCount+1
            predtictionMatrix[j].append(1)
        else:
            predtictionMatrix[j].append(0)
    
    trainingAccuracy = float(tempCount*100)/len(predtictionMatrix)
    
    return trainingAccuracy


    
def calculateConfusion(testKNN, classVal):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    totalOccurance = 0

    for i in range(len(testKNN)):
        if (testKNN[i][1] == classVal):
            if (testKNN[i][3] == False):
                tp = tp +1
            else:
               fn = fn + 1
        else:
            if (testKNN[i][3] == False):
                fp = fp + 1
            else:
                tn = tn+1

    tpr = tp/float(tp+fn)
    fpr = fp/float(fp+tn)

    return tpr, fpr

trainingImageMaster = fetchImage('train-images.idx3-ubyte')
trainingLabelMaster = fetchLabel('train-labels.idx1-ubyte')

classOne = fetchClass(trainingLabelMaster, 200, 1)
classTwo = fetchClass(trainingLabelMaster, 200, 2)
classSeven = fetchClass(trainingLabelMaster, 200, 7)

applicableList = classOne+classTwo+classSeven

trainingImage, trainingLabel = extractTrainingData(trainingImageMaster, trainingLabelMaster, applicableList)

predictedKnnMatrix = performCrossValidation(trainingImage, trainingLabel)

NearestNeighbour(trainingImage, trainingLabel)

best_model(predictedKnnMatrix)

print "Best Model is:",bestModel,"-NN"
print "Accuracy while training is:", trainingAccuracy

testImage = []
testLabel = []

for i in range(50):
    indx = randint(0, 199)
    testImage.append(trainingImageMaster[classOne[indx]])
    testLabel.append(trainingLabelMaster[classOne[indx]])

for i in range(50):
    indx = randint(0, 199)
    testImage.append(trainingImageMaster[classTwo[indx]])
    testLabel.append(trainingLabelMaster[classTwo[indx]])
    
for i in range(50):
    indx = randint(0, 199)
    testImage.append(trainingImageMaster[classSeven[indx]])
    testLabel.append(trainingLabelMaster[classSeven[indx]])
    
predictedKnnMatrix = []

for i in range(len(testLabel)):
    tempKnn = []
    tempKnn.append(predictTestDigit(testImage[i]))
    tempKnn.append(testLabel[i])
    tempKnn.append(i)
    predictedKnnMatrix.append(tempKnn)

#predictedKnnMatrix contain [predicted, correct, index, correct/incorrect]
print "Accuracy while testing is:", accuracy(predictedKnnMatrix)

objects = ('1 TPR', '1 FPR', '2 TPR', '2 FPR', '7 TPR', '7 FPR')
y_pos = np.arange(len(objects))
truePositiveRate, falsePositiveRate = calculateConfusion(predictedKnnMatrix, 1)
print "TP for Class 1:", truePositiveRate
print "FP for Class 1:", falsePositiveRate

performance = [truePositiveRate, falsePositiveRate]

truePositiveRate, falsePositiveRate = calculateConfusion(predictedKnnMatrix, 2)
print "TP for Class 2:", truePositiveRate 
print "FP for Class 2:", falsePositiveRate

performance.append(truePositiveRate)
performance.append(falsePositiveRate)

truePositiveRate, falsePositiveRate = calculateConfusion(predictedKnnMatrix, 7)
print "TP for Class 7:", truePositiveRate
print "FP for Class 7:", falsePositiveRate

performance.append(truePositiveRate)
performance.append(falsePositiveRate)


plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Value in Percentage')
plt.title('FP/FN on basis of class')

