#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# In[ ]:


def load_data(path):
    lookup = dict()
    reverselookup = dict()
    count = 0
    #For-loop to read each label
    for j in os.listdir(r'leapGestRecog/00'):
        #To avoid hidden folders and files
        if not j.startswith('.'):
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1
    
    x_data = []
    y_data = []
    #Tallying images
    datacount = 0
    #Looping over the ten top-level folders
    for i in range(0, 10): 
        for j in os.listdir(r'leapGestRecog/0' + str(i) + '/'):
            if not j.startswith('.'):
                #Tally images of each gesture
                count = 0
                for k in os.listdir(r'leapGestRecog/0' + str(i) + '/' + j + '/'):
                    path = r'leapGestRecog/0' + str(i) + '/' + j + '/' + k
                    #Reading images in grayscale
                    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    #Resizing
                    img = cv2.resize(img, (150,150))
                    #Turning them into numpy arrays
                    arr = np.array(img)
                    x_data.append(arr) 
                    count = count + 1
                y_values = np.full((count, 1), lookup[j]) 
                y_data.append(y_values)
                datacount = datacount + count
    x_data = np.array(x_data, dtype = 'float32')
    y_data = np.array(y_data)
    y_data = y_data.reshape(datacount, 1)
    
    #Visualising one sample of each gesture
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))          
    ax = axes.ravel()
    for i in range(0, 10): 
        ax[i].imshow(x_data[i*200 , :, :])
        ax[i].title.set_text(reverselookup[y_data[i*200 ,0]])
    plt.subplots_adjust(hspace=0.5) 
    plt.show()
    
    return x_data, y_data

