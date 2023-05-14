#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


# In[3]:


lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir(r'leapGestRecog/00'):
    if not j.startswith('.'): #To avoid hidden folders and files
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
lookup


# In[4]:


x_data = []
y_data = []
datacount = 0 #Image tally
for i in range(0, 10): # Looping over the ten top-level folders
    for j in os.listdir(r'leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'):
            count = 0 # Tally images of each gesture
            for k in os.listdir(r'leapGestRecog/0' + str(i) + '/' + j + '/'):
                path = r'leapGestRecog/0' + str(i) + '/' + j + '/' + k
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) #Reading images in grayscale
                img = cv2.resize(img, (150,150)) #Resizing them to a 150x150 square
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1)


# In[5]:


fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))          
ax = axes.ravel()
for i in range(0, 10):
    ax[i].imshow(x_data[i*200 , :, :])
    ax[i].title.set_text(reverselookup[y_data[i*200 ,0]])
plt.subplots_adjust(hspace=0.5) 
plt.show()


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 0.25)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


# In[7]:


def preprocess(img,label):
    img=img/255
    img=tf.expand_dims(img, axis=-1)
    data_preprocessed=(img,label)
    return data_preprocessed


# In[8]:


batch_size = 128
train_dataset = train_dataset.map(preprocess)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess)
test_dataset = test_dataset.batch(batch_size)


# In[9]:


num_classes = 10
model = keras.Sequential([
    layers.Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (150, 150, 1)),
    layers.MaxPooling2D(pool_size =(2, 2)),
    layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'),
    layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
    layers.Conv2D(filters = 96, kernel_size = (3, 3), padding = 'Same', activation = 'relu'),
    layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
    layers.Conv2D(filters = 96, kernel_size = (3, 3), padding = 'Same', activation = 'relu'),
    layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
    layers.Flatten(),
    layers.Dense(512),
    layers.Activation('relu'),
    layers.Dense(10, activation = 'softmax'),
    ])


# In[10]:


model.summary()


# In[11]:


loss="sparse_categorical_crossentropy"
metrics = ["accuracy"]
optimizer=tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[12]:


epoch = 10
history = model.fit(train_dataset, batch_size = 128, epochs=epoch, verbose=1, validation_data=test_dataset)


# In[13]:


[loss, acc] = model.evaluate(test_dataset,verbose=1, batch_size=128)
print("Accuracy:" + str(acc))


# In[14]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[15]:


model.save('hand_recognition_model.h5')


# In[16]:


import sys 
from PIL import Image
model = keras.models.load_model('hand_recognition_model.h5')
video = cv2.VideoCapture(0)


# In[17]:


while True:
    _, frame = video.read()
    kernel = np.ones((3,3),np.uint8)
    cv2.imshow("Capturing", frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#Define range of skin color in HSV
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
#Extract skin colour image
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
#Extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask,kernel,iterations = 4)
#Blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100)
    mask = cv2.resize(mask,(150,150))
    img_array = np.array(mask)
#Changing dimension from 150x150 to 150x150x1
    img_array = np.stack((img_array,)*1, axis=-1)
#Changing dimension from 150x150x1 into 1x150x150x1 
    img_array_ex = np.expand_dims(img_array, axis=0)
    # print(img_array_ex.shape)
#Calling the predict method on model to predict gesture in the frame
    prediction = model.predict(img_array_ex, verbose=0)
    predicted_class = np.argmax(prediction)
    finalprediction = 'Processing'
    if predicted_class == 0:
        finalprediction = 'Palm'
    elif predicted_class == 1:
        finalprediction = 'L shape'
    elif predicted_class == 2:
        finalprediction = 'Fist'
    elif predicted_class == 3:
        finalprediction = 'Fist (side)'
    elif predicted_class == 4:
        finalprediction = 'Thumb'
    elif predicted_class == 5:
        finalprediction = 'Index'
    elif predicted_class == 6:
        finalprediction = 'Ok Sign'
    elif predicted_class == 7:
        finalprediction = 'Palm (side)'
    elif predicted_class == 8:
        finalprediction = 'Curved Hand'
    elif predicted_class == 9:
        finalprediction = 'Down'
    else:
        finalprediction = 'Processing'
    key=cv2.waitKey(1) 
    if key == ord('q'):
        break 
    if key == ord('p'):
        hand_image = frame
        hand_image = cv2.resize(hand_image,(150,150))
        text = finalprediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        hand_image = cv2.putText(hand_image, text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Hand Gesture', hand_image)
    # if key == ord('j'):
    #     cv2.imshow('output', mask)
    # if key == ord('k'):
        # cv2.destroyWindow('output')
        # cv2.destroyWindow('Hand Gesture')
video.release()
cv2.destroyAllWindows()

