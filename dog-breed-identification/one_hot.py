import numpy as np

# One hot maker

naming_dict = {} # id: breed
f = open("classes_t.csv", "r")
fileContents = f.read()
fileContents = fileContents.split('\n')
one_hot = np.zeros(121, np.int8)
one_hot[120] = 1

for i in range(len(fileContents)-1):
  fileContents[i] = fileContents[i].split(',')
  naming_dict[fileContents[i][0]] = fileContents[i][0]
  inter = fileContents[i][0]
  print('elif word_label == '+ inter +' : return np.' + repr(np.roll(one_hot, -i)).replace(', dtype=int8',''))
