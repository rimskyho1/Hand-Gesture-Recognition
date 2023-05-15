#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


# In[ ]:


#Function to preprocess the data
def preprocessing(img,label):
    img=img/255
    img=tf.expand_dims(img, axis=-1)
    data_preprocessed=(img,label)
    return data_preprocessed


# In[ ]:


#Function to split it into train and test data, then apply the preprocess function and batching them
def data_split(x, y, batch_size): 
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.map(preprocessing)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.map(preprocessing)
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset

