from PIL import Image
import numpy as np
import os
import csv
from random import shuffle
import imageio
import matplotlib.pyplot as plt
from sklearn import preprocessing

DIR = './train'

naming_dict = {} # id: breed
f = open("labels.csv", "r")
fileContents = f.read()
fileContents = fileContents.split('\n')
for i in range(len(fileContents)-1):
  fileContents[i] = fileContents[i].split(',')
  naming_dict[fileContents[i][0]] = fileContents[i][1]

breeds = naming_dict.values()
breed_set = set(breeds)
counting_dict = {}
for i in breed_set:
  counting_dict[i] = 0

#print(breeds)


print(breed_set)

# for img in os.listdir(DIR):
#     imgName = img.split('.')[0] # converts '0913209.jpg' --> '0913209'
#     label = naming_dict[str(imgName)]
#     counting_dict[label] += 1
#     path = os.path.join(DIR, img)
#     saveName = './labeled_train/' + label + '-' + str(counting_dict[label]) + '.jpg'
#     image_data = np.array(Image.open(path))
#     imageio.imwrite(saveName, image_data)

def label_img(name):
    word_label = name.split('-')[0]
    if word_label == 'golden_retriever' : return np.array([0, 0, 0, 0, 0, 0, 0, 1])
