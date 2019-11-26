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

# augmentation for nvidia model and attemption to generlise to Track 2
def augment_brightness_camera_images(image):
    image_adjusted = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image_adjusted = np.array(image_adjusted, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image_adjusted[:,:,2] = image_adjusted[:,:,2]*random_bright
    image_adjusted[:,:,2][image_adjusted[:,:,2]>255]  = 255
    image_adjusted = np.array(image_adjusted, dtype = np.uint8)
    image_adjusted = cv2.cvtColor(image_adjusted,cv2.COLOR_HSV2RGB)
    return image_adjusted

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
            
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


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


# data argumentation
augmented_images, augmented_measurements = [], []


## to fix left bias since when training most of the time the car turning left
# genrate more new right turn iamges by flip all left turn images to right turn images 
for image, measurement in zip(images_train, measurements_train):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        
        augmented_images.append(augment_brightness_camera_images(image))
        augmented_measurements.append(measurement)
        
        augmented_images.append(add_random_shadow(image))
        augmented_measurements.append(measurement)
        
        r = np.random.uniform() # randomly flip data with 0.5 percent chance
        if r > .5:
            augmented_images.append(cv2.flip(image,1))
            augmented_measurements.append(measurement*-1.0)
        

# convert to numpy for Keras
## TODO: change back
augmented_images = np.array(augmented_images)
augmented_measurements = np.array(augmented_measurements)

images_val = np.array(images_val)
measurements_val = np.array(measurements_val)



# Graph
model_nvidia = Sequential()

# Data Preprocessing
## normalization -255
## mean centering -0.5 
model_nvidia.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))

# cropping image
model_nvidia.add(Cropping2D(cropping=((65,25), (0,0))))

#nvidia
model_nvidia.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model_nvidia.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model_nvidia.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model_nvidia.add(Convolution2D(64,3,3,activation="relu"))
model_nvidia.add(Convolution2D(64,3,3,activation="relu"))
model_nvidia.add(Flatten())
model_nvidia.add(Dense(100))
model_nvidia.add(Dense(50))
model_nvidia.add(Dense(10))
model_nvidia.add(Dense(1))

# train the model as a regression network
model_nvidia.compile(loss='mse', optimizer='adam')
history_object = model_nvidia.fit(augmented_images, augmented_measurements, nb_epoch=4, verbose=1, validation_data=(images_val, measurements_val), shuffle=True)

# save the model for later test on simulator
model_nvidia.save('model_nvidia.h5')

exist()