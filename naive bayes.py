import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, isdir, exists
import tarfile
import pickle

cifar10_dataset_folder_path = 'cifar-10-batches-py'

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
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
    training_accuracies = np.zeros(5)
    testing_accuracies = np.zeros(5)
    training_f1 = np.zeros(5)
    testing_f1 = np.zeros(5)
    training_time = np.zeros(5)
    testing_time = np.zeros(5)
    for i in range(1,6) :
        if i == 1 :
            print('Performing analysis for train batch : 1')
        elif i == 5 :
            print('Performing analysis for all the batches')
        else :
            print('Performing analysis for the first {} train batches'.format(i))
        
        start = time()
        clf.fit(train_X[:(i*10000)],train_y[:(i*10000)])
        end = time()
        training_time[i-1] = end-start
        
        pred_train_y = clf.predict(train_X[:(i*10000)])
        
        start = time()
        pred_test_y = clf.predict(test_X)
        end = time()
        testing_time[i-1] = end-start
        
        training_accuracies[i-1] = accuracy_score(pred_train_y,train_y[:(i*10000)])
        testing_accuracies[i-1] = accuracy_score(pred_test_y,test_y)
        print("Accuracy on train set when trained on {} images is : {}".format((i*10000),training_accuracies[i-1]))
        print("Accuracy on test set when trained on {} images is : {}".format((i*10000),testing_accuracies[i-1]))
        
        training_f1[i-1] = f1_score(pred_train_y,train_y[:(i*10000)],average='weighted')
        testing_f1[i-1] = f1_score(pred_test_y,test_y,average='weighted')
        print("f1 on train set when trained on {} images is : {}".format((i*10000),training_f1[i-1]))
        print("f1 on test set when trained on {} images is : {}".format((i*10000),testing_f1[i-1]))
        print()
    
    fig, ax = plt.subplots(ncols=3, figsize = (17,5))
    ind = np.arange(5)
    bar_width = 0.45
    colors = ['#A00000','#00A0A0']
    
    ax[0].bar(ind, training_time , width = bar_width, color = colors[0],label='training')
    ax[0].bar(ind+bar_width, testing_time , width = bar_width, color = colors[1],label='testing')
    
    ax[1].bar(ind, training_accuracies , width = bar_width, color = colors[0],label='training')
    ax[1].bar(ind+bar_width, testing_accuracies , width = bar_width, color = colors[1],label='testing')
    
    ax[2].bar(ind, training_f1 , width = bar_width, color = colors[0],label='training')
    ax[2].bar(ind+bar_width, testing_f1 , width = bar_width, color = colors[1],label='testing')
    
    for j in range(3) :
        ax[j].set_xticks(ind + bar_width / 2)
        ax[j].set_xticklabels(('B1', 'B2', 'B3', 'B4', 'B5'))
        #ax[0, j].legend((rects1[0], rects2[0]), ('training', 'testing'))
    
    ax[0].set_ylabel("Time (in seconds)")
    ax[1].set_ylabel("Accuracy Score")
    ax[2].set_ylabel("F-score")
    
    ax[0].set_title("Model Training")
    ax[1].set_title("Accuracy Score")
    ax[2].set_title("F-score")
    
    plt.show()
    
    
    from sklearn.metrics import confusion_matrix
    cm_train = confusion_matrix(labels, pred_train_y)
    cm_test = confusion_matrix(labels_test, pred_test_y)
    
    print()
    print()
    print()
    print('Confusion matrix for training data')
    for i in range(10):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm_train[i, :], class_name)

    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))
    
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
    
    
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

print('-------------------------------------Performing analysis on colored data-------------------------------------')
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
print('-------------------------------Performing analysis on normalized colored data-------------------------------')
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
print('------------------------------------Performing analysis on gray-scale data------------------------------------')
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
print('-------------------------------Performing analysis on normalized gray-scale data-------------------------------')
print()
print()
perform_analysis(clf,normalize(features_gray).reshape(50000,32*32),labels,normalize(features_test_gray).reshape(10000,32*32),labels_test)