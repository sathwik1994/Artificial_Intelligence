import numpy as np

def fetchImage(fileName):
    training_images_file = open(fileName,'rb') 
    training_images = training_images_file.read()
    training_images_file.close()

    training_images = bytearray(training_images)
    training_images = training_images[16:]

    image_array = np.array(training_images)
    image_array[image_array<=230] = 0
    image_array[image_array>230] = 1
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

def trainNaive(image, label):
    pixelCount = [[0 for x in range(28*28)] for y in range(10)] 
    valueCount = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]            
    
    xIndx = 0
    for i in range(0, 60000):
        xIndx = label[i]
        valueCount[xIndx][0] = valueCount[xIndx][0]+1.0
        pixelCount[xIndx] = pixelCount[xIndx]+image[i]
    
    
    pixelCount = np.array(pixelCount)
    valueCount = np.array(valueCount)
    pixelProbabilty = pixelCount/valueCount
    digitProbability = valueCount/60000
    return digitProbability, pixelProbabilty
 
def maxListIndex(a):
    max = a[0]
    maxIndex = 0
    
    for i in range(len(a)):
        if a[i] > max:
            max = a[i]
            maxIndex = i
    return maxIndex, max


def calculateDifference(actualLabel, predictedLabel):
    count = 0
    for i in range(10000):
        if actualLabel[i] != predictedLabel[i]:
            count = count+1
            
    return count
    
def predictImage(trainedDigitProbability, trainedPixelProbabilty, testImage):
    nonZeroIndex = []
    probableDigit = [1 for x in range(10)]
    for i in range(0, 28*28):
        if testImage[i] == 1:
            nonZeroIndex.append(i)
    
    for indx in nonZeroIndex:
        for j in range(0, 10):
            probableDigit[j] = probableDigit[j]*(trainedPixelProbabilty[j][indx]/trainedDigitProbability[j][0])
    
    maxIndex, max = maxListIndex(probableDigit)
    return maxIndex
                
trainingImage = fetchImage('train-images.idx3-ubyte')
trainingLabel = fetchLabel('train-labels.idx1-ubyte')

testImage = fetchImage('t10k-images.idx3-ubyte')
testActualLabel = fetchLabel('t10k-labels.idx1-ubyte')
predictedLabel = []

trainedDigitProbability, trainedPixelProbabilty = trainNaive(trainingImage, trainingLabel)


for i in range(10000):
    predictedLabel.append(predictImage(trainedDigitProbability, trainedPixelProbabilty, testImage[i]))

errCount = calculateDifference(testActualLabel, predictedLabel)
accuracyPercent = ((10000-errCount)*100)/10000.0
print "Accuracy with Naive Bayes is: ", accuracyPercent, "%"
