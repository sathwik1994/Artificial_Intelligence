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
def one_hot_encode(x):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(n_values=10)    
    one_hot_encoded_labels = enc.fit_transform(np.array(x).reshape(-1, 1)).toarray()
    return one_hot_encoded_labels

def normalize(x):
    normalized_x = np.zeros(tuple(x.shape))
    nr_images = x.shape[0]
    max_val, min_val = x.max(), x.min()
    for image_index in range(nr_images):
        normalized_x[image_index,...] = (x[image_index, ...] - float(min_val)) / float(max_val - min_val)    
    return normalized_x

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))
    

def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        validation_count = int(len(features) * 0.1)

        # Prprocess and save a batch of training data
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            features[:-validation_count],
            labels[:-validation_count],
            'preprocess_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation.p')

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the training data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all training data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_training.p')
    
    
preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

import tensorflow as tf

def neural_net_image_input(image_shape):
    return tf.placeholder(tf.float32, shape=((None,) + image_shape), name='x')
    
def neural_net_label_input(n_classes):
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')

def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, shape=(None), name='keep_prob')

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    weights_shape = list(conv_ksize) + [x_tensor.get_shape().as_list()[3], conv_num_outputs]
    weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=5e-2))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    
    output = tf.nn.conv2d(x_tensor, weights, 
                          strides=[1, conv_strides[0], conv_strides[1], 1],
                          padding='SAME')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)
    
    output = tf.nn.max_pool(output, 
                            ksize=[1, pool_ksize[0], pool_ksize[1], 1], 
                            strides=[1, pool_strides[0], pool_strides[1], 1],
                            padding='SAME')                                  
    return output

def flatten(x_tensor):
    tensor_shape = x_tensor.get_shape().as_list()
    flattened_shape = np.array(tensor_shape[1:]).prod()
    return tf.reshape(x_tensor, [tf.shape(x_tensor)[0], flattened_shape])

def fully_conn(x_tensor, num_outputs):
    flattened_shape = np.array(x_tensor.get_shape().as_list()[1:]).prod()
    weights = tf.Variable(tf.truncated_normal([flattened_shape, num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros([num_outputs]))
    fc = tf.nn.relu(tf.add(tf.matmul(x_tensor, weights), bias))        
    return fc

def output(x_tensor, num_outputs):
    flattened_shape = np.array(x_tensor.get_shape().as_list()[1:]).prod()
    weights = tf.Variable(tf.truncated_normal([flattened_shape, num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros([num_outputs]))
    return tf.add(tf.matmul(x_tensor, weights), bias)

def conv_net(x, keep_prob):
    conv = conv2d_maxpool(x,
                           conv_num_outputs=64,
                           conv_ksize=[5,5],
                           conv_strides=[1,1],
                           pool_ksize=[3,3],
                           pool_strides=[2,2])
    
    conv = conv2d_maxpool(conv,
                          conv_num_outputs=64,
                          conv_ksize=[5,5],
                          conv_strides=[1,1],
                          pool_ksize=[3,3],
                          pool_strides=[2,2])
    
    # Apply a Flatten Layer
    flattened_conv = flatten(conv)
    
    # 2 Fully-Connected Layers.
    fc = fully_conn(flattened_conv, 768)
    fc = fully_conn(fc, 192)
    
    # Dropout layer.
    fc = tf.nn.dropout(fc, keep_prob)
    
    # Output Layer.
    return output(fc, 10)      

tf.reset_default_graph()
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()
logits = conv_net(x, keep_prob)
logits = tf.identity(logits, name='logits')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, feed_dict={x: feature_batch,
                                      y: label_batch,
                                      keep_prob: keep_probability})
    
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    loss = sess.run(cost, feed_dict={x: feature_batch,
                                     y: label_batch,
                                     keep_prob: 1.})
    valid_acc = sess.run(accuracy, feed_dict={x: valid_features,
                                              y: valid_labels,
                                              keep_prob: 1.})

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
    
def batch_features_labels(features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size):
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    return batch_features_labels(features, labels, batch_size)

epochs = 30
batch_size = 256
keep_probability = 0.75

save_model_path = './image_classification'
print('Training...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
        
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
    
import random

try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})

        
test_model()
 