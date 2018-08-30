import numpy as np
import math
from random import randint
import matplotlib.pyplot as plt
from PIL import Image
np.set_printoptions(threshold=np.nan)

def generateImage(image_array):
    k = 0
    image_matrix_sample = [[0 for x in range(28)] for y in range(28)] 
    
    for i in range (0, 28):
        for j in range (0, 28):
            image_matrix_sample[i][j] = image_array[k]
            k = k+1
    
    image_matrix_sample = np.asarray(image_matrix_sample)

    img = Image.fromarray(image_matrix_sample)
    img.save("my.png")
    img.show()

def fetchImage(fileName):
    training_images_file = open(fileName,'rb') 
    training_images = training_images_file.read()
    training_images_file.close()

    training_images = bytearray(training_images)
    training_images = training_images[16:]

    image_array = np.array(training_images)
    image_array = np.reshape(image_array, (-1, 28*28))
    
    return image_array


def fetchLabel(fileName):
    training_label_file = open(fileName,'rb') 
    training_label = training_label_file.read()
    training_label_file.close()

    training_label = bytearray(training_label)
    training_label = training_label[8:]

    label_array = np.array(training_label)
    
    return label_array

def fetchFive(trainingLabel, requiredCount):
    fives = []
    
    count = 0
    while (count < requiredCount):
        indx = randint(0, 60000-1)
        if trainingLabel[indx] == 5:
            fives.append(indx)
            count = count+1
    
    return fives

def fetchNonFive(trainingLabel, requiredCount):
    nonFives = []
    
    count = 0
    while (count < requiredCount):
        indx = randint(0, 60000-1)
        if trainingLabel[indx] != 5:
            nonFives.append(indx)
            count = count+1
    
    return nonFives

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

    
def calculateMean(trainingImage, trainingLabel):
    meanArray = [[0 for x in range(28*28)] for y in range(10)]
    classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(trainingLabel)):
            meanArray[trainingLabel[i]] = meanArray[trainingLabel[i]]+trainingImage[i]
            classCount[trainingLabel[i]] = classCount[trainingLabel[i]]+1
    
    for i in range(len(meanArray)):
        meanArray[i] = meanArray[i]/float(classCount[i])
    
    meanArray = np.array(meanArray)
    return meanArray
    

def calculateSdSquare(trainingImage, trainingLabel, meanArray):
    sdsArray = [[0 for x in range(28*28)] for y in range(10)]
    classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    for i in range(len(trainingLabel)):
        tempNumpy = trainingImage[i]-meanArray[trainingLabel[i]]
        tempNumpy = np.square(tempNumpy)
        sdsArray[trainingLabel[i]] = sdsArray[trainingLabel[i]]+tempNumpy
        classCount[trainingLabel[i]] = classCount[trainingLabel[i]]+1
        
        
    for i in range(len(sdsArray)):
        sdsArray[i] = sdsArray[i]/float(classCount[i])
        
    sdsArray = np.array(sdsArray)
    return sdsArray
    
    
def trainGaussian(trainingImage, trainingLabel):
    meanMatrix = calculateMean(trainingImage, trainingLabel)
    sdsMatrix = calculateSdSquare(trainingImage, trainingLabel, meanMatrix)
    return meanMatrix, sdsMatrix
    

def predictImage(meanMatrix, sdsMatrix, testImage):
    validIndex = []
    
    for i in range(len(testImage)):
        if testImage[i] > 0 :
            validIndex.append(i)
            
    probabilityMatrix = [[0 for x in range(len(validIndex))] for y in range(10)]
    
            
    for i in range(len(validIndex)):
        for j in range(len(sdsMatrix)):
            if sdsMatrix[j][validIndex[i]] == 0:
                probabilityMatrix[j][i] = 0
            else:
                meanDev = testImage[i] - meanMatrix[j][validIndex[i]] 
                meanDev = (meanDev**2)/float(sdsMatrix[j][validIndex[i]])
                probabilityMatrix[j][i] = math.sqrt(2*math.pi*sdsMatrix[j][validIndex[i]])*math.exp(meanDev*(-0.5))
                
    prod = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    for j in range(len(prod)):
        for i in probabilityMatrix[j]:
            prod[j] = prod[j]*i
        
    maxInd = 0
    maxVal = prod[0]

    for i in range(len(prod)):
        if prod[i]>maxVal:
            maxInd = i
            maxVal = prod[i]
        
    if maxInd == 5:
        return 1
    else:
        return 0

def countValue(lst, val):
    count = 0
    for v in lst:
        if v == val:
            count = count+1
    return count

def calculateConfusion(predictedLabel):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(predictedLabel)):
        if (predictedLabel[i][1] == 1):
            if (predictedLabel[i][0] == predictedLabel[i][1]):
                tp = tp+1
            else:
                fn = fn + 1
        else:
            if (predictedLabel[i][0] == predictedLabel[i][1]):
                tn = tn+1
            else:
                fp = fp + 1
            
    print "TP:", tp
    print "FP:", fp
    print "TN:", tn
    print "FN:", fn
    
    fpr = fp/float(fp+tn)
    fnr = fn/float(tp+fn)
    
    
    return fpr, fnr
            

    
trainingImage = fetchImage('train-images.idx3-ubyte')
trainingLabel = fetchLabel('train-labels.idx1-ubyte')


onlyFives = fetchFive(trainingLabel, 1000)
exceptFives = fetchNonFive(trainingLabel, 1000)

applicableList = onlyFives+exceptFives


trainingImage, trainingLabel = extractTrainingData(trainingImage, trainingLabel, applicableList)


testImage = trainingImage[:200, :]
testLabel = trainingLabel[:200]
predictedLabel = []

trainingImage = trainingImage[200:, :]
trainingLabel = trainingLabel[200:]

meanMatrix, sdsMatrix = trainGaussian(trainingImage, trainingLabel)


for i in range(len(testLabel)):
    tempPredicted = []
    tempPredicted.append(predictImage(meanMatrix, sdsMatrix, testImage[i]))
    if testLabel[i] == 5:
        tempPredicted.append(1)
    else:
        tempPredicted.append(0)
    predictedLabel.append(tempPredicted)
    

type1, type2 = calculateConfusion(predictedLabel)
print "Type 1:", type1
print "Type 2:", type2
type1List = [5*type1, 2*type1, type1, type1, type1]
type2List = [type2, type2, type2, 2*type2, 5*type2]
plt.plot(type1List, type2List)
plt.show() 

