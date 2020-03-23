# One hot maker

naming_dict = {} # id: breed
f = open("classes_t.csv", "r")
fileContents = f.read()
fileContents = fileContents.split('\n')

for i in range(len(fileContents)-1):
  fileContents[i] = fileContents[i].split(',')
  naming_dict[fileContents[i][0]] = fileContents[i][0]
  inter = fileContents[i][0]
  print('elif word_label == '+ inter +' : return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])')