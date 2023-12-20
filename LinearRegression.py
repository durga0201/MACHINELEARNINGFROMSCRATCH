import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 



class LinearRegression():
    def __init__(self,learning_rate,iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    
    def fit(self,X,Y):


