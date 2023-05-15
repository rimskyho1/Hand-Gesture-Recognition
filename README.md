This repository contains the following files and folders:
- Dataloader.py
- Hand Gesture Recognition.py
- leapGestRecog
- List of Hand Gestures.png
- Model_Test.py
- Model_Training.py
- Preprocessor.py
- README.md
- requirements.txt


Instructions:
- Install the modules in requirements.txt
- Run Hand Gesture Recognition.py and let it read the dataset, train a model, and open the webcam to test the model
- Once the capturing window is active, you can make one of the 10 predefined hand gestures (refer to "List of Hand Gestures.png")
- Works best if your hand is the only visible part of your body and is centred
- Press "p" to have it make a multiclass classification prediction using the trained model (can be pressed multiple times to repeatedly make new predictions)
- If not accurate, try readjusting your hand around in different angles
- Press "q" to end the OpenCV process

requirements.txt contains Python modules required for this script.

leapGestRecog contains the dataset used for training and testing. It is a Kaggle-based dataset categorised into 10 folders named 00, 01, 02...up to 09. Each of these folders represent one individual subject (person) who provided the hand gestures for this project. Inside each of these folders, you will find another set of subfolders which are labelled based on images of which hand gesture it contains, with 10 total. Each hand gesture folder contains a set of 200 near-infrared hand gesture images. The total dataset contains 20,000 images. 

List of Hand Gestures.png is a Matplotlib-generated image showcasing all 10 hand gestures and their labels. 

Dataloader.py contains a function to read the dataset as numpy arrays.

Preprocessor.py contains a function to preprocess the dataset and split it into train and test sets.

Model_Training.py contains a function to create and train a model based on the td.data.Dataset.

Model_Test.py contains a function to load the model, open the webcam with OpenCV, and make predictions with it.

Hand Gesture Recognition.py contains a code to import and run all of the aforementioned scripts. 

