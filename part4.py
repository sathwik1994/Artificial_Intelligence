#!/usr/bin/python
import numpy as np
from scipy import optimize

def flower_to_float(s):
    d = {'Iris-setosa':0.,
         'Iris-versicolor':1.,
         'Iris-virginica':2.
        }
    return d[s]


# Read iris data
data = np.loadtxt("iris.data",delimiter=',',converters={4:flower_to_float})

# Shuffle data
np.random.seed(0)
np.random.shuffle(data)

K   = 3 # No of class
N,_ = data.shape # No of rows

# Split label and data
l = data[:,4]

# Convert label into one-hot array
label = np.zeros([N,K])
for j in range(K):
    label[:,j] = np.array(l==j,dtype=np.float)

# Set leftmost column to 1 (bias)
datas = np.zeros([N,5])
datas[:,0]  = 1.
datas[:,1:] = data[:,0:4] 

# Split to training and test data
train_datas = datas[:N/2,:]
train_label = label[:N/2,:] 

trial_datas = datas[N/2:,:]
trial_label = label[N/2:,:] 

# Function for gradient descent
def f(w):
    alpha = 0.01
    s = 0.
    for n,xn in enumerate(train_datas):
        tn = train_label[n]
        s1,s2 = 0.,0.
        for j in range(K):s1 += tn[j]*np.dot(w[j*5:(j+1)*5],xn)
        for j in range(K):s2 += np.exp(np.dot(w[j*5:(j+1)*5],xn))
        s += s1-np.log(s2)
    y = 0.5*alpha*np.dot(w.transpose(),w) - s
    return y

w_init = np.ones(15)
w_hat  = optimize.minimize(f,w_init)

w = w_hat.x

# Classify using softmap
score = 0
for n,xn in enumerate(trial_datas):
    tn = trial_label[n]

    # Calculate softmap
    s = np.zeros([3])
    for j in range(K): s[j] = np.exp(np.dot(w[j*5:(j+1)*5],xn))
    s = s/np.sum(s)
    
    # Get the highest probability
    predict = np.argmax(s)
    correct = np.argmax(trial_label[n])
    print xn,predict,correct
    if predict==correct:score+=1

print "Score: %.2lf%%" % (float(score)/len(trial_datas)*100)
