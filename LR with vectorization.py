# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:29:17 2019

@author: Asus
"""


import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import numpy as np
x1 = np.array([10,11,12,13,14,15,16,17,18,19,20])
x2 = np.array([110,111,220,222,333,444,555,440,220,660,440])
x = np.array([x1,x2])
x = scale(x,axis=1)
y = np.array([1,1,1,1,1,1,0,0,0,0,0])

m = np.zeros((2,1))
c = 1
dm = 0
dc = 0
alpha = 0.01
e = 0

le = []
import time
for i in range(0,5000):
    #Linear combination
    z = np.matmul(x.T , m) + c
    #Non linear transformation (Sigmoid)
    y_cap = 1/(1+np.e**(-z))

    #Calculating derivatives     
    diff = (y_cap - y.reshape(-1,1))
    dm =  (x.T * diff).sum(axis=0)
    dc = np.sum((y_cap.T - y))
    
    #Calculating cross entropy loss
    celoss =  -np.mean(y*np.log(y_cap).T + (1-y)*np.log(1-y_cap).T)
    le.append(celoss)
    
    #Updating coefficients and intercept
    m = m - alpha * dm.reshape(-1,1)
    c = c - alpha * dc

plt.plot(le)

y_cap = y_cap.flatten()
for i in range(0,len(y_cap)):
    if y_cap[i] < 0.5:
        y_cap[i] = 0
    else:
        y_cap[i] = 1

#Accuracy
accuracy = (y==y_cap).sum()/ len(y)




