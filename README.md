This repository contains the following files and folders:
Hand Gesture Recognition.py
leapGestRecog
List of Hand Gestures.png
README.md
requirements.txt

requirements.txt contains Python modules required for this script.

leapGestRecog contains the dataset used for training and testing. It is a Kaggle-based dataset categorised into 10 folders named 00, 01, 02...up to 09. Each of these folders represent one individual subject (person) who provided the hand gestures for this project. Inside each of these folders, you will find another set of subfolders which are labelled based on images of which hand gesture it contains, with 10 total. Each hand gesture folder contains a set of 200 near-infrared hand gesture images. The total dataset contains 20,000 images. 

List of Hand Gestures.png is a Matplotlib-generated image showcasing all 10 hand gestures and their labels. 

Hand Gesture Recognition.py is a script that does the following:
- Reads all of these images as numpy arrays
- Splits them into test and training sets
- Converts them into TensorFlow tf.data.Dataset format
- Preprocesses and batches the data
- Trains a sequential model on the data using Keras 
- Evaluates and visualises the results
- Saves the model as "hand_recognition_model.h5" in the same directory
- Uses OpenCV to open a capturing window with a camera
- Loads "hand_recognition_model.h5"
- Once the capturing window is active, you can make one of the 10 predefined hand gestures (refer to "List of Hand Gestures.png")
- Works best if your hand is the only visible part of your body and is centred
- Press "h" to have it make a multiclass classification prediction using the trained model (can be pressed multiple times to repeatedly make new predictions)
- If not accurate, try readjusting your hand around in different angles
- Press "q" to end the OpenCV process
