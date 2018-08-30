import numpy as np
import matplotlib.pyplot as plt

training_images_file = open('train-images-idx3-ubyte','rb')
training_images = training_images_file.read()
training_images_file.close()

training_images = bytearray(training_images)

training_images = training_images[16:]

train_X = np.asarray(training_images)

train_X = np.reshape(train_X, (60000,784))

first_train = np.reshape(train_X[0:1],(28,28))

plt.imshow(first_train)
plt.show()

training_labels_file = open('train-labels-idx1-ubyte','rb')
train_y = training_labels_file.read()
training_labels_file.close()

train_y = bytearray(train_y)

train_y = train_y[8:]

train_y = np.asarray(train_y)

print " The train element printed above is : ",train_y[0]

train_y = np.reshape(train_y,(60000,1))

index_5 = np.argwhere(train_y == 5)[:,0]
np.random.shuffle(index_5)
train_5 = np.array(train_X[index_5[0:900]],dtype = float)
test_5 = np.array(train_X[index_5[-100:]],dtype = float)

index_not_5 = np.argwhere(train_y !=5)[:,0]
np.random.shuffle(index_not_5)
train_not_5 = np.array(train_X[index_not_5[0:900]],dtype = float)
test_not_5 = np.array(train_X[index_not_5[-100:]],dtype = float)

test_X = np.concatenate((test_5, test_not_5), axis=0)
test_y = np.concatenate((np.ones(100),np.zeros(100)), axis=0)

mean_5 = np.mean(train_5,axis=0)
mean_not_5 = np.mean(train_not_5,axis=0)

std_5 = np.sqrt(np.mean(abs(train_5 - mean_5)**2,axis=0))
std_not_5 = np.sqrt(np.mean(abs(train_not_5 - mean_not_5)**2,axis=0))
"""
zero_std_5=np.argwhere(std_5==0)
zero_std_not_5=np.argwhere(std_not_5==0)

drop_index = np.unique(np.concatenate((zero_std_5,zero_std_not_5),0))

train_5 = np.delete(train_5,[drop_index],axis=1)
train_not_5 = np.delete(train_not_5,[drop_index],axis=1)
test_X = np.delete(test_X,[drop_index],axis=1)

std_5 = np.delete(std_5,[drop_index])
std_not_5 = np.delete(std_not_5,[drop_index])

mean_5 = np.delete(mean_5,[drop_index])
mean_not_5 = np.delete(mean_not_5,[drop_index])
"""
#if std_5 = 0, data_5 = 1
data_5 = train_5.copy()
for i in range(len(data_5)) :
    for j in range(len(data_5[i])) :
        if std_5[j] ==0 :
            data_5[i,j] = 1
        else :
            data_5[i,j] = np.exp((-(train_5[i,j]-mean_5[j])**2)/2*std_5[j]**2)/np.sqrt(2*np.pi*(std_5[j]**2))
            

data_not_5 = train_not_5.copy()
for i in range(len(data_not_5)) :
    for j in range(len(data_not_5[i])) :
        if std_not_5[j] ==0 :
            data_not_5[i,j] = 1
        else :
            data_not_5[i,j] = np.exp((-(train_not_5[i,j]-mean_not_5[j])**2)/2*std_not_5[j]**2)/np.sqrt(2*np.pi*(std_not_5[j]**2))
"""
data_5 = np.exp((-(train_5-mean_5)**2)/2*std_5**2)/np.sqrt(2*np.pi*(std_5**2))
data_not_5 = np.exp((-(train_not_5-mean_not_5)**2)/2*std_not_5**2)/np.sqrt(2*np.pi*(std_not_5**2))
"""
prob_5 = np.array(sum(data_5),dtype=float)
prob_5 = prob_5/sum(prob_5)
prob_not_5 = np.array(sum(data_not_5),dtype=float)
prob_not_5 = prob_not_5/sum(prob_not_5)

prob_x1y1 = train_5.copy()
for i in range(len(prob_x1y1)) :
    for j in range(len(prob_x1y1[i])) :
        if std_5[j] ==0 :
            prob_x1y1[i,j] = 1
        else :
            prob_x1y1[i,j] = np.exp((-(data_5[i,j]-mean_5[j])**2)/2*std_5[j]**2)/np.sqrt(2*np.pi*(std_5[j]**2))
            

prob_x1y0 = train_not_5.copy()
for i in range(len(prob_x1y0)) :
    for j in range(len(prob_x1y0[i])) :
        if std_not_5[j] ==0 :
            prob_x1y0[i,j] = 1
        else :
            prob_x1y0[i,j] = np.exp((-(data_not_5[i,j]-mean_not_5[j])**2)/2*std_not_5[j]**2)/np.sqrt(2*np.pi*(std_not_5[j]**2))
"""
prob_x1y1 = np.exp(-(data_5-mean_5)**2/2*std_5**2)/np.sqrt(2*np.pi*std_5**2)
prob_x1y0 = np.exp(-(data_not_5-mean_not_5)**2/2*std_not_5**2)/np.sqrt(2*np.pi*std_not_5**2)

prob_x1y1 = data_5
prob_x1y0 = data_not_5
"""

pred_y = np.zeros(200)
for i in range(0,len(test_X)) :
    x = test_X[i:i+1]*prob_x1y1
    x[x == 0] =1
    x = np.prod(x)*0.5
    
    y = test_X[i:i+1]*prob_x1y0
    y[y == 0] =1
    y = np.prod(y)*0.5
    
    if x > y :
        pred_y[i] = 1
            
correct = 0
true_positive = 0
true_negative = 0
false_negative = 0
false_positive = 0

for i in range(0,len(pred_y)) :
    if pred_y[i] == test_y[i] :
        correct = correct+1
        
for i in range(0,len(pred_y)) :
    if pred_y[i] == 1 :
        if test_y[i] == 1 :
            true_positive = true_positive + 1
        else :
            false_positive = false_positive + 1
    else :
        if test_y[i] == 1 :
            false_negative = false_negative + 1
        else :
            true_negative = true_negative + 1
print ""
print "Out of 200 samples",correct,"samples were predicted correctly, returning an accuracy of",float(correct)/2.0,"."
print (true_positive+true_negative)