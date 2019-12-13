# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:29:17 2019

@author: Asus
"""


import matplotlib.pyplot as plt
import numpy as np
x = [10,11,12,13,14,15,16,17,18,19,20]
y = [1,1,1,1,1,1,0,0,0,0,0]

plt.scatter(x,y)

m = 0
c = 0
dm = 0
dc = 0
alpha = 0.0005
e = 0

le = []
import time
for i in range(0,100000):
    #time.sleep(0.2)
    e = 0
    dm = 0
    dc = 0
    for j in range(0, len(x)):
        z = m * x[j] + c
        #print('z = ',z)
        y_cap = 1/(1+np.e**(-z))
        #print('ycap = ',y_cap)
        celoss +=  (y[j]*np.log(y_cap) + (1-y[j])*np.log(1-y_cap))
        dm += x[j] * (y_cap - y[j])
        dc += (y_cap - y[j])
    print(celoss)
    le.append(-celoss/len(y))
    m = m - alpha * dm
    c = c - alpha * dc

plt.plot(le)

y_c = []
for i in range(0,len(y)):
    if 1/(1+np.e**(-(m * x[i] + c))) < 0.5:
        y_c.append(0)
    else:
        y_c.append(1)

#Accuracy
accuracy = (np.array(y)==np.array(y_c)).sum()/ len(y)
