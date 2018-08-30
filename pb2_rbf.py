## LINEAR REGRESSION USING RADIAL BASIS FUNCTIONS ##
##############################################
import numpy as np
from matplotlib import pyplot as plt


def mlrms_error(true_data,pred_data):
    Nt = len(true_data)
    error = np.subtract(true_data, pred_data)
    rms_error = np.sqrt(np.divide(np.sum(error**2), Nt))
    return rms_error
def gaussian(loca,mean,stdev):
    val = np.exp(-np.divide(np.square(np.subtract(loca,mean)),2*np.square(stdev)))
    return val

#############################################
data = np.loadtxt('C:/Users/LSAdmin/Desktop/crash.txt')
train_data = data[0::2]
test_data  = data[1::2]
print('train_data',train_data[0])
figure1 = plt.figure(1)
plt.plot(data[:,0],data[:,1],'k')
plt.plot(train_data[:,0],train_data[:,1],'ob')
plt.plot(test_data[:,0],test_data[:,1],'or')
plt.draw
plt.savefig('crash_data.png',bbox_inches='tight')
##############################################
Np = 5
L = np.linspace(5,25,Np)
Nt = len(train_data[:,0])
Nte = len(test_data[:,0])
error_train = np.zeros((Np,1))
error_test = np.zeros((Np,1))
W = np.zeros((26,26))
p = 0
for j in range(0,Np):
    stdev = np.divide(60,L[j])
    mean = np.linspace(stdev - 0.5*stdev,L[j]*stdev - 0.5*stdev,np.int64(L[j]))
    Nw = np.int64(L[j])
    phi = np.zeros((Nt, Nw))
    phi_test = np.zeros((Nte, Nw))
    for k in range(0,Nw):
        phi[:,k] = gaussian(train_data[:,0],mean[k],stdev)
        phi_test[:,k] = gaussian(test_data[:,0],mean[k],stdev)
    A = np.matmul(phi.T,phi)
    B = np.matmul(phi.T ,train_data[:,1])
    nw = B.shape[0]
    W[p,:nw] = np.linalg.solve(A,B)
    pred_train = np.matmul(phi,W[p,:nw])
    pred_test  = np.matmul(phi_test,W[p,:nw])
    error_train[p] = mlrms_error(train_data[:,1],pred_train)
    error_test[p] = mlrms_error(test_data[:, 1],pred_test)
    p = p + 1
figure2 = plt.figure(2)
plt.plot(L,error_train,'ob')
plt.plot(L,error_test,'or')
plt.xlim(1,30)
plt.draw

plt.savefig('error_MLRMS_gaussian.png',bbox_inches='tight')

best_model_train = np.argmin(error_train)
best_model_test = np.argmin(error_test)
print('The best model with minimum training error is',L[best_model_train],'gaussians')
print('The best model with minimum test error is',L[best_model_test],'gaussians')
Nnew = 100
X_in = np.linspace(0,60,Nnew)
phi_train = np.zeros((Nnew,np.int64(L[best_model_train])))
phi_test = np.zeros((Nnew,np.int64(L[best_model_test])))
#
stdev_train = np.divide(60,np.int64(L[best_model_train]))
stdev_test = np.divide(60,np.int64(L[best_model_test]))
mean_train = np.linspace(stdev_train - 0.5 * stdev_train, L[best_model_train] * stdev_train - 0.5 * stdev_train, np.int64(L[best_model_train]))
mean_test = np.linspace(stdev_test - 0.5 * stdev_test, L[best_model_test] * stdev_test - 0.5 * stdev_test, np.int64(L[best_model_test]))
Nw_train = np.int64(L[best_model_train])
Nw_test = np.int64(L[best_model_test])

for k in range(0,Nw_train):
    phi_train[:,k] = gaussian(X_in,mean_train[k],stdev_train)

for k in range(0,Nw_test):
    phi_test[:,k] = gaussian(X_in,mean_test[k],stdev_test)
#
pred_train = np.matmul(phi_train,W[best_model_train,:np.int64(L[best_model_train])])
pred_test  = np.matmul(phi_test,W[best_model_test,:np.int64(L[best_model_test])])

figure3 = plt.figure(3)
plt.plot(data[:,0],data[:,1],'k')
plt.plot(X_in,pred_train,'ob')
plt.plot(X_in,pred_test,'or')
plt.ylim(-180,100)
plt.draw
plt.savefig('final_pred_gaussian.png',bbox_inches='tight')
