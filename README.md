# MYO_GestureClassifier
MAST(Myo Armband Sign-Language Translator): This project implements the classfication of 10 medical sign language gestures using Random Forest. The data of the gesture is represented by EMG signals gathered by the MYO Sensor.

## "docs" Directory:
This directory includes the survey, proposal and the final results of this project.

## "Sample_RandomForest" Directory:
This directoy has a python code for implementing the RandomForest from scratch. Moreover, it has a simple comparison between SKLEARN RandomForest API accuracy and the one coded by us. (The dataset used for evaluation can be found here: [dataset](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009))

## "gesture_classifier" Directory:
This directory is the main directory that includes a python code to connect to the MYO sensor, train the random forest and test the final trained gestures".

## "CNN_Comparison" Directory:
This directory includes a pytorch notebook that implements a custom RESNet26 with SENet which classifies the same 10 gestures. However, the data is images instead of EMG signals. The goal of this CNN is to compare the accuracy with our MAST approach.
