# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:29:17 2019

@author: Asus
"""

from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target
x = x[0:99]
x = x.T[0:99]
y = y[0:99]

m = np.zeros((4,1))
c = 1

dm = 0
dc = 0
alpha = 0.001
e = 0

le = []
import time
for i in range(0,1000):
    #time.sleep(0.2)
    z = np.matmul(x.T , m) + c
    y_cap = 1/(1+np.e**(-z))
    diff = (y_cap - y.reshape(-1,1))
    dm =  (x.T * diff).sum(axis=0)
    dc = np.sum( (y_cap.T - y))
    
    celoss = -np.mean(y*np.log(y_cap).T + (1-y)*np.log(1-y_cap).T)
    le.append(celoss)
    
    m = m - alpha * dm.reshape(-1,1)
    c = c - alpha * dc
    y_cap_plot = []

plt.plot(le)
print(le[-1])
y_c = []
for i,j in zip(y_cap,y):
    if i < 0.5:
        y_c.append(0)
        print(0 , '    ' , j)
    else:
        y_c.append(1)
        print(1, '     ' , j)

#Accuracy
accuracy = (y==y_c).sum()/ len(y)
