import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, isdir, exists
import tarfile
import pickle

cifar10_dataset_folder_path = 'cifar-10-batches-py'

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()


#tests.test_folder_path(cifar10_dataset_folder_path)
train_files = [cifar10_dataset_folder_path + '/data_batch_' + str(batch_id) for batch_id in range(1, 6)]
other_files = [cifar10_dataset_folder_path + '/batches.meta', cifar10_dataset_folder_path + '/test_batch']
missing_files = [path for path in train_files + other_files if not exists(path)]

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

with open(cifar10_dataset_folder_path+'/test_batch', mode='rb') as file:
    batch = pickle.load(file, encoding='latin1')

features_test = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
labels_test = batch['labels']

features, labels = load_cfar10_batch(cifar10_dataset_folder_path, 1)
for i in range(2,6) :
    features_i, labels_i = load_cfar10_batch(cifar10_dataset_folder_path, i)
    features = np.concatenate([features,features_i],axis=0)
    labels = np.concatenate([labels,labels_i],axis=0)
    
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

features_test_gray = np.zeros((10000,32,32))
features_gray = np.zeros((50000,32,32))
for i in range(len(features_test)) :
    features_test_gray[i] = rgb2gray(features_test[i])
for i in range(len(features)) :
    features_gray[i] = rgb2gray(features[i])
    
from sklearn.metrics import accuracy_score,f1_score
from time import time
def perform_analysis(clf,train_X,train_y,test_X,test_y) :
    pred_test_y = np.zeros((10,10000,10))
    
    testing_accuracies = np.zeros(11)
    
    testing_f1 = np.zeros(11)
    
    training_time = np.zeros(11)
    
    testing_time = np.zeros(11)
    
    for i in range(1,11) :
        print('Performing analysis on train batches : {} part : {}'.format((i+1)//2,((i-1)%2)+1))
        
        start = time()
        clf.fit(train_X[((i-1)*5000):(i*5000)],train_y[((i-1)*5000):(i*5000)])
        end = time()
        training_time[i-1] = end-start
        
        start = time()
        pred_y = clf.predict(test_X)
        end = time()
        testing_time[i-1] = end-start
        
        for index in range(len(pred_y)) :
            pred_test_y[i-1,index,pred_y[index]] = 1
        
        testing_accuracies[i-1] = accuracy_score(pred_y,test_y)
        print("Accuracy on test set : {}".format(testing_accuracies[i-1]))
        
        testing_f1[i-1] = f1_score(pred_y,test_y,average='weighted')
        print("f1 on test set : {}".format(testing_f1[i-1]))
        print()
    
    
    training_time[10] = np.sum(training_time)
    
    testing_time[10] = np.sum(testing_time)
    
    pred = np.sum(pred_test_y,axis=0)
    pred_y_ensemble = np.zeros(10000)
    for i in range(10000) :
        pred_y_ensemble[i] = np.argmax(pred[i])
        
    testing_accuracies[10] = accuracy_score(pred_y_ensemble,test_y)
    print("Accuracy of ensemble SVMs is : {}".format(testing_accuracies[10]))
    
    testing_f1[10] = f1_score(pred_y_ensemble,test_y, average='weighted')
    print("F1 of ensembles SVMs is : {}".format(testing_f1[10]))
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (16,9))
    ind = np.arange(11)
    bar_width = 0.70
    colors = ['#A00000','#00A0A0']
    
    ax[0,0].bar(ind, training_time , width = bar_width, color = colors[0],label='training time')
    
    ax[0,1].bar(ind, testing_time , width = bar_width, color = colors[1],label='testing time')
    
    ax[1,0].bar(ind, testing_accuracies , width = bar_width, color = colors[1],label='testing accuracy')
    
    ax[1,1].bar(ind, testing_f1 , width = bar_width, color = colors[0],label='testing f1')
    
    for a in range(2) :
        for b in range(2) :
            ax[a,b].set_xticks(ind + bar_width / 2)
            ax[a,b].set_xticklabels(('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'ALL'))
    
    ax[0,0].set_ylabel("Training Time (in seconds)")
    ax[0,1].set_ylabel("Testing Time (in seconds)")
    ax[1,0].set_ylabel("Accuracy Score")
    ax[1,1].set_ylabel("F-score")
    
    ax[0,0].set_title("Model Training")
    ax[0,1].set_title("Model Testing")
    ax[1,0].set_title("Accuracy Score")
    ax[1,1].set_title("F-score")
    
    plt.show()
    
    
    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(labels_test, pred_y_ensemble)
    
    print()
    print()
    print('Confusion matrix for testing data')
    for i in range(10):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm_test[i, :], class_name)

    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))
    
    
    
def normalize(x):
    normalized_x = np.zeros(tuple(x.shape))
    nr_images = x.shape[0]
    # Compute max/min values.
    max_val, min_val = x.max(), x.min()
    # Transform every image.
    for image_index in range(nr_images):
        normalized_x[image_index,...] = (x[image_index, ...] - float(min_val)) / float(max_val - min_val)    
    return normalized_x
    
from sklearn.svm import SVC
clf = SVC()


print('-------------------------------Performing analysis on colored data-------------------------------')
print()
print()
perform_analysis(clf,features.reshape(50000,32*32*3),labels,features_test.reshape(10000,32*32*3),labels_test)

print()
print()
print()
print()
print()
print()
print()
print()
print('--------------------------Performing analysis on normalized colored data--------------------------')
print()
print()
perform_analysis(clf,normalize(features).reshape(50000,32*32*3),labels,normalize(features_test).reshape(10000,32*32*3),labels_test)
print()
print()
print()
print()
print()
print()
print()
print()
print('-------------------------------Performing analysis on gray-scale data-------------------------------')
print()
print()
perform_analysis(clf,features_gray.reshape(50000,32*32),labels,features_test_gray.reshape(10000,32*32),labels_test)
print()
print()
print()
print()
print()
print()
print()
print()
print('--------------------------Performing analysis on normalized gray-scale data--------------------------')
print()
print()
perform_analysis(clf,normalize(features_gray).reshape(50000,32*32),labels,normalize(features_test_gray).reshape(10000,32*32),labels_test)