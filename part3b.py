#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import sys

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

# Get the size of training and test data
N1, = train_data.x.shape
N2, = trial_data.x.shape

# Set the radial basis count
#L = [5,10,15,20,25]
l     = 50
beta  = 0.0025
alpha = np.logspace(-8,0,100) 

# Calculate the parameters for Gaussian function
sigma = 60./l
mu = np.linspace(sigma - 0.5*sigma,l*sigma - 0.5*sigma,l)
k = -1./(2.*sigma**2)

rmserror = []
for i,a in enumerate(alpha):

    # Calculate phi for training data
    phi = np.zeros([N1,l])
    for j in range(l): phi[:,j] = np.exp(k*(train_data.x-mu[j])**2)
       
    # Calculate w
    w = np.linalg.solve(np.dot(phi.T,phi) + (a/beta)*np.identity(l),np.dot(phi.T,train_data.y))

    # Calculate rms
    train_rms = np.sqrt(np.mean((train_data.y - np.dot(phi,w))**2))

    # Repeat for trial data
    phi = np.zeros([N2,l])
    for j in range(l): phi[:,j] = np.exp(k*(trial_data.x-mu[j])**2)
    trial_rms = np.sqrt(np.mean((trial_data.y - np.dot(phi,w))**2))

    # Append RMS and W for reference later
    rmserror.append([a,train_rms,trial_rms,w])


# Print out the calculated RMS
print "Alpha train_rms trial_rms"
for a,train_rms,trial_rms,w in rmserror:
    print a,train_rms,trial_rms

# Plot RMS
plt.xlabel("Radial function count")
plt.ylabel("RMS error")
plt.semilogy(alpha,[y[1] for y in rmserror],label="Training data")
plt.semilogy(alpha,[y[2] for y in rmserror],label="Test data")
plt.legend(loc=2)
plt.grid(True)
plt.show()

# Find the best count for training and test data
train_best = min(rmserror,key=lambda k:k[1])
trial_best = min(rmserror,key=lambda k:k[2])

print "Best alpha for training data: %.3lf" % train_best[0]
print "Best alpha for test data: %.3lf" % trial_best[0]

# Re-calculate x fitting
datax = np.linspace(0,60,101)

# Calculate phi
sigma = 60./l
mu = np.linspace(sigma - 0.5*sigma,l*sigma - 0.5*sigma,l)
k = -1./(2.*sigma**2)
phi = np.zeros([101,l])
for j in range(l): phi[:,j] = np.exp(k*(datax-mu[j])**2)


# Plot data
plt.xlabel("Time [ms]")
plt.ylabel("Acceleration")
plt.plot(data[:,0],data[:,1],'k')
plt.plot(datax,np.dot(phi,train_best[3]),'ob',label="Training data,alpha=%.3lf" % train_best[0])
plt.plot(datax,np.dot(phi,trial_best[3]),'og',label="Test data,alpha=%.3lf" % trial_best[0])

plt.legend(loc=2)
plt.title("Training data")
plt.grid(True)
plt.xlim([0,60])
plt.ylim([-150,100])
plt.show()
