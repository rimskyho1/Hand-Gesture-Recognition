#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


# In[ ]:


def train(train_input, validation_input, epoch):
    #Create model
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
    
    #Compile model
    loss="sparse_categorical_crossentropy"
    metrics = ["accuracy"]
    optimizer=tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics) 
    
    #Fit model
    history = model.fit(train_input, batch_size = 128, epochs=epoch, verbose=1, validation_data=validation_input)

    #Evaluate model
    [loss, acc] = model.evaluate(validation_input, verbose=1, batch_size=128)
    print("Accuracy:" + str(acc))
    
    #Visualise evaluation
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
    
    #Save model
    model.save('hand_recognition_model.h5')

