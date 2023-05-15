#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing basic modules necessary for many operations
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# In[2]:


#Importing all modules related to our deep learning model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


# In[3]:


#Importing modules for testing on webcam
import sys 
from PIL import Image


# In[4]:


#Importing the Python scripts of this project
import Dataloader
import Preprocessor
import Model_Training


# In[5]:


#Loading the data
path = r'leapGestRecog/00'
x_data, y_data = Dataloader.load_data(path)


# In[6]:


#Splitting the data
batch_size = 128
train_dataset, test_dataset = Preprocessor.data_split(x_data, y_data, 128)


# In[7]:


#Creating and training the model
train_input = train_dataset
validation_input = test_dataset
epoch = 10
Model_Training.train(train_input, validation_input, epoch)


# In[8]:


#Loads the model and tests it with webcam
import Model_Test

