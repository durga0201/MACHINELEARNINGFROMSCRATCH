import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 

class LinearRegression_GD():
    def __init__(self,learning_rate=0.01,iterations=10000,eps=1e-6):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.eps = eps
    
    def predict(self,X):
        return np.dot(X,self.w.T)
    
    def cost(self,X,y):
        y_pred = self.predict(X)
        loss = (y-y_pred)**2
        return np.mean(loss)
    
    def grad(self,X,y):
        y_pred = self.predict(X)
        d_intercept = -2*np.sum(y-y_pred)
        d_x = -2*sum(X[:,1:] * (y-y_pred).reshape(-1,1))
        g = np.append(np.array(d_intercept),d_x)
        return g/X.shape[0]
    
    def fit(self,X,y,method="standard",verbose =True):
        self.w = np.zeros(X.shape[1])
        w_hist = [self.w]
        cost_hist = [self.cost(X,y)]

        for iter in range(self.iterations):
            g = self.grad(X,y)
            if method =="standard":
                step = self.learning_rate*g
            else:
                raise ValueError("method not defined")
            self.w = self.w - step
            w_hist.append(self.w)

            J = self.cost(X,y)
            cost_hist.append(J)

            if verbose:
                print(f"Iter :{iter},gradient:{g},params:{self.w},cost:{J}")

            if np.linalg.norm(w_hist[-1]-w_hist[-2])<self.eps:
                break
        
        self.iteration = iter +1 
        self.w_hist = w_hist
        self.cost_hist = cost_hist
        self.method = method

        return self
     

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([2,4,6,8,10,12,14,16])
LinearRegression_GD().fit(X, y, "standard")


class LinearRegression_BGD():
    def __init__(self, learning_rate=0.01, iterations=10000, eps=1e-6, batch_size=2):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.eps = eps
        self.batch_size = batch_size
    
    def predict(self, X):
        return np.dot(X, self.w.T)
    
    def cost(self, X, y):
        y_pred = self.predict(X)
        loss = (y - y_pred) ** 2
        return np.mean(loss)
    
    def grad(self, X_batch, y_batch):
        y_pred = self.predict(X_batch)
        d_intercept = -2 * np.sum(y_batch - y_pred)
        d_x = -2 * np.sum(X_batch[:, 1:] * (y_batch - y_pred).reshape(-1, 1), axis=0)
        g = np.append(np.array(d_intercept), d_x)
        return g / X_batch.shape[0]
    
    def fit(self, X, y, method="mini-batch", verbose=True):
        self.w = np.zeros(X.shape[1])
        w_hist = [self.w]
        cost_hist = [self.cost(X, y)]

        n_batches = int(np.ceil(X.shape[0] / self.batch_size))

        for iter in range(self.iterations):
            for batch in range(n_batches):
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, X.shape[0])
                X_batch = X[start:end, :]
                y_batch = y[start:end]

                g = self.grad(X_batch, y_batch)
                if method == "mini-batch":
                    step = self.learning_rate * g
                else:
                    raise ValueError("Method not defined")

                self.w = self.w - step
                w_hist.append(self.w)

            J = self.cost(X, y)
            cost_hist.append(J)

            if verbose:
                print(f"Iter :{iter}, params:{self.w}, cost:{J}")

            if np.linalg.norm(w_hist[-1] - w_hist[-2]) < self.eps:
                break
        
        self.iteration = iter + 1
        self.w_hist = w_hist
        self.cost_hist = cost_hist
        self.method = method

        return self

# Example usage with mini-batch gradient descent
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16])

LinearRegression_BGD().fit(X, y, "mini-batch")