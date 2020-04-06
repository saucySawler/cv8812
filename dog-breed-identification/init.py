#################################################################################################
#
# Project		: Dog Breed Classifier
# Program Name		: init.py
# Authors		: Sakif Mohammed and Clayton Sawler
# Course		: Engineering 8814 - Computer Vision
# Date			: April 7, 2020
# Purpose		: Loads a trained model to classify images of dogs into breeds.
#
#################################################################################################

from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import csv
from random import shuffle

# Load images

directory = './labeled_train'
test_data = []
for img in os.listdir(directory):
    path = os.path.join(directory, img)
    if "DS_Store" not in path:
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((300, 300), Image.ANTIALIAS)
        test_data.append(np.array(img))
shuffle(test_data)
images = np.array(test_data).reshape(-1, 300, 300, 1)

# Load model

yaml_file = open('classifier.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = tf.keras.models.model_from_yaml(loaded_model_yaml)

loaded_model.load_weights("classifier.h5")
print("Loaded model from disk.")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# Load classes

with open('classes.csv', newline='') as f:
    reader = csv.reader(f)
    breeds = list(reader)

# Predict classes

classes = loaded_model.predict_classes(images)

i = 0
for img in os.listdir(directory):
    print(img)
    print(breeds[0][classes[i]])
    i = i+1
