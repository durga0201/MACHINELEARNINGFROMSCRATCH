import numpy as np


def local_weighted_regression(x0,X,Y,tau):
    #add bias term
    x0 = np.r_[1,x0]
    X = np.c_[np.ones(len(X)),X]

    xw =    X.T*weight_calculate(x0,X,tau)
    theta = np.linalg.pinv(xw @ X) @ xw @ Y

    return x0 @ theta

def weight_calculate(x0,X,tau):
    return np.exp(np.sum((X-x0)**2,axis=1)/(-2*(tau**2)))



tau = 1.0

#define distribution
n = 1000
 
# generate dataset
X = np.linspace(-3, 3, num=n)
Y = np.abs(X ** 3 - 1)
 
# jitter X
X += np.random.normal(scale=.1, size=n)

domain = np.linspace(-3,3,num=3000)
prediction = [local_weighted_regression(x0,X,Y,tau) for x0 in domain]