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

breed_list = list(breed_set)
le = preprocessing.LabelEncoder()
le.fit(breed_list)
print(le.classes_)
breed_tf = le.transform(breed_list)
print(breed_tf)