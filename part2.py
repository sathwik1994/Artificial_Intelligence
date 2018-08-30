#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

# Class for data matrix
class Data():
    def __init__(self,data):
        self.x = data[:,0]
        self.y = data[:,1]


# Read data
data = np.loadtxt("crash.txt")

# Split test-training
train_data = Data(data[0::2]) # Even rows
trial_data = Data(data[1::2]) # Odd rows

N1, = train_data.x.shape
N2, = trial_data.x.shape

L = [5,10,15,20,25]

rmserror = []
for i,l in enumerate(L):

    mu = np.linspace(0,60,l)
    sigma = 60./(l-1)
    k = -1./(2.*sigma**2)

    # Calculate phi
    phi = np.zeros([N1,l])
    for j in range(l): phi[:,j] = np.exp(k*(train_data.x-mu[j])**2)
        
    # Calculate w
    w_train = np.linalg.solve(np.dot(phi.transpose(),phi),np.dot(phi.transpose(),train_data.y))

    # Calculate rms
    train_rms = np.sqrt(np.mean((train_data.y - np.dot(phi,w_train))**2))

    # Repeat for trial data
    phi = np.zeros([N2,l])
    for j in range(l): phi[:,j] = np.exp(k*(trial_data.x-mu[j])**2)
    w_trial = np.linalg.solve(np.dot(phi.transpose(),phi),np.dot(phi.transpose(),trial_data.y))
    trial_rms = np.sqrt(np.mean((trial_data.y - np.dot(phi,w_trial))**2))

    rmserror.append([l,train_rms,w_train,trial_rms,w_trial])


print "Function_count train_rms trial_rms"
for p,train_rms,t1,trial_rms,t2 in rmserror:
    print p,train_rms,trial_rms

# Plot RMS
plt.xlabel("Radial function count")
plt.ylabel("RMS error")

plt.semilogy(L,[y[1] for y in rmserror],label="Training data")
plt.semilogy(L,[y[3] for y in rmserror],label="Test data")
plt.legend(loc=2)
plt.grid(True)
plt.show()

# Find the best count for training and test data
train_best = min(rmserror,key=lambda k:k[1])
trial_best = min(rmserror,key=lambda k:k[3])

print "Best function count for training data: %d" % train_best[0]
print "Best function count for test data: %d" % trial_best[0]

# Re-calculate fitting
train_x = np.linspace(np.min(train_data.x),np.max(train_data.x),201)
trial_x = np.linspace(np.min(trial_data.x),np.max(trial_data.x),201)

# Calculate phi
mu = np.linspace(0,60,train_best[0])
sigma = 60./(train_best[0]-1)
k = -1./(2.*sigma**2)
train_phi = np.zeros([201,train_best[0]])
for j in range(l): train_phi[:,j] = np.exp(k*(train_x-mu[j])**2)

# Calculate phi
mu = np.linspace(0,60,trial_best[0])
sigma = 60./(trial_best[0]-1)
k = -1./(2.*sigma**2)
trial_phi = np.zeros([201,trial_best[0]])
for j in range(l): trial_phi[:,j] = np.exp(k*(trial_x-mu[j])**2)

# Plot training data
plt.xlabel("Time [ms]")
plt.ylabel("Acceleration")
plt.plot(train_data.x,train_data.y,'o',label="Actual data")
plt.plot(train_x,np.dot(train_phi,train_best[2]),label="Best radial basis function fit,L=%s" % train_best[0])
plt.legend(loc=2)
plt.title("Training data")
plt.grid(True)
plt.xlim([0,60])
plt.ylim([-150,100])
plt.show()

# Plot trial data
plt.xlabel("Time [ms]")
plt.ylabel("Acceleration")
plt.plot(trial_data.x,trial_data.y,'o',label="Actual data")
plt.plot(trial_x,np.dot(trial_phi,trial_best[2]),label="Best radial basis function fit,L=%s" % trial_best[0])
plt.legend(loc=2)
plt.title("Training data")
plt.grid(True)
plt.xlim([0,60])
plt.ylim([-150,100])
plt.show()

