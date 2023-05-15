#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import os

import tensorflow as tf
from tensorflow import keras

import sys 
from PIL import Image


# In[ ]:


def video_capture(model_path):
    #Loading the model
    model = keras.models.load_model(model_path)
    #Initiating video capture with OpenCV
    video = cv2.VideoCapture(0)
    
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
    #Break the whole process if the 'q' key is pressed
        if key == ord('q'):
            break 
    #Make a prediction if the 'p' key is pressed
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
    video.release()
    cv2.destroyAllWindows()


# In[ ]:


load_model = r'hand_recognition_model.h5'
video_capture(load_model)

