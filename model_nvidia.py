import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


lines = []
images = []
measurements = []

# read rows in driving_log.csv
with open('newdata/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  
  for line in reader:
      lines.append(line)

# read images and steering measurements 
for line in lines:
  # create adjusted steering measurements for the side camera images
  correction = 0.20 # this is a parameter to tune
  measurement_center = float(line[3])
  measurement_left = measurement_center + correction
  measurement_right = measurement_center - correction
  
  
  # line is the row of the data set    
  #local_path = 'D:\\OneDrive\\Documents\\SDCN\\Term 1\\Behavioral Cloning\\udacitydata\\IMG\\'
  local_path = ''
  source_path_center = line[0].split('/')[-1]
  source_path_left = line[1].split('/')[-1]
  source_path_right = line[2].split('/')[-1]

  curr_path_center = local_path + source_path_center
  curr_path_left = local_path + source_path_left
  curr_path_right = local_path + source_path_right
  

  img_center = cv2.imread(curr_path_center)
  img_left = cv2.imread(curr_path_left)
  img_right = cv2.imread(curr_path_right)
  
  # convert image back to RGB since cv2 read image in BGR format
  img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
  img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
  img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
  
  # Image as feature
  images.append(img_center)
  images.append(img_left)
  images.append(img_right)

  # measurement as labels
  measurements.append(measurement_center)
  measurements.append(measurement_left)
  measurements.append(measurement_right)


# Divide data into training and validation set
images_train, images_val, measurements_train, measurements_val = train_test_split(images, measurements, test_size=0.3, random_state=0)
        

# convert to numpy for Keras
images_train = np.array(images_train)
measurements_train = np.array(measurements_train)

images_val = np.array(images_val)
measurements_val = np.array(measurements_val)




# ## Train a Network

# # Graph
model = Sequential()

# Data Preprocessing
## normalization -255
## mean centering -0.5 
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))

# cropping image
model.add(Cropping2D(cropping=((50,20), (0,0))))

#LeNet
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


# train the model as a regression network
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(images_train, measurements_train, nb_epoch=5, verbose=1, validation_data=(images_val, measurements_val), shuffle=True)

# save the model for later test on simulator
model.save('model_augmented_brightness.h5')

exist()