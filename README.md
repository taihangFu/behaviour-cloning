
# **Behavioral Cloning** 
---
**Device**
* Windows 10
* 16Gb Ram
* GeForce GTX 1070

[//]: # (Image References)

[imbalance_distribution]: ./class_distribution_with_center_camera_only.png
[capture]: ./Capture.PNG
[lenet]: ./lenet.png
[nvidia]: ./cnn-architecture-624x890.png

### Submission Related
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the LENET model
* **model_nvidia.py** containing the script to create and train the NVIDIA model
* **drive.py** for driving the car in autonomous mode
* **model_and_run.ipynb** just for reference
* **model.h5** containing a trained convolution neural network with LeNet architecture
* **model_nvidia.h5** containing another trained convolution neural network Nvidia architecture
* **writeup_report.md** summarizing the results
* **video.mp4** recording for vehicle wiht LENET model to drive on Track 1
* **video_nvidia.mp4** recording for vehicle wiht NVIDIA model to drive on Track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
**OR**
```sh
python drive.py model_nvidia.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Overview on Model Architecture and Training Strategy
#### 1. An appropriate model architecture has been employed
I create 2 models with differnt purposes
* **Lenet** achitecture(model.h5)
* **Nvidia** achitecture(model_nvidia.h5). 

* Both succesfully drive autonomously on Track 1 and both fail on Track 2.

* while the model with Nvidia achitecture perform slighly better on Track 2.

Further details on next section.

#### 2. Attempts to reduce overfitting in the model

My approach to reduce overfitting is simply just adding more augmented data. Since the model did quite well with only augumented data and without any regulization techniques such as Dropout.

Further details for data augmentation on next section.

#### 3. Model parameter tuning

The hyper parameter for tuning are mainly epochs. And after mulitples attemptions, I found the best is epoches 3-5.

I applied adam optimizor instead of custom learning rate.

I tried splitting in different ratio for training and validation set and I found 0.7:0.3 is the best in my case.

Another parameters such as how much to crop the image to reduce backgound noise, the steering angle measurement correction for images from left and right cameras. Multiple tuning of these paremeter applied as in my case they are the keys leads to the vehicle finally drive on Track 1 successfully.  

Further details on next section.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. It mainly involves,
for both model.h5 and model_nvidia.h5:
* center lane driving
* flipping data to increase mroe right turn
* images from left and right cameras with custome steering angle offset for the purpose of recovery back to the center lane

extra data for model_nvidia.h5:
* augmented images with random shadow 
* augmented images with random brightness 

For details about how I created the training data, see the next section. 

###  Architecture and Training Strategy

#### 1. Solution Design Approach

I divided the approach into 2 phrases:
Phrase 1 is for driving autonomously on Track 1 with simlpier **LeNet** model namely **model.h5**
Phrase 2 is for attemption on generalising to Track 2 with more complicated model, nvidia architecture and more data augmentation of track one data only namely **model_nvidia.h5**

**Phrase 1**
My first step was to use a pure LeNet5 on the training data I collected, which consisted on 2 laps of track one. I simply want to see how well it works with no data preparation and no data processing or hyper parameter tuning. 

It turns out it cannot even drive property on the first 4 seconds, it just keep doing left turn so I guess it could be due to the imbalance distribution of classes for the data since most of the curve are **left** curve on track 1.

![imbalance_distribution]

In order to solve the imbalance issue, I did the proper data augumentation that will be mentioned on the below "3. Creation of the Training Set & Training Process", which in brief, flip the data so as to create more right turn 


Then I also crop the images(data) to reduce noises as David in lecture videos suggestes. 

The final step is to do some data preprocessing such as Normalization as mentioned on "3. Creation of the Training Set & Training Process" below to speed up the training process and save memory.

I try The result turns out to be huge difference, it could successfully make half lap **until it reach**

![capture]


The car **go stright outside the lane instead on truning left...** 
To improve the driving behavior in these cases, I add extra images from both left and right and add steering angle offset of 0.2 to create a left and right recovery, but it turns out it behave the same as before, **it still NOT able to revognise the left curved** on above image.

I struggled for a while and later I try to **adjust the cropping size** by **reducing the amount of cropping** since the curve on the image is bit difference from another curve, it has a way for going striaght to the lake, so I thought maybe I could keep some more of the background view so the model could recognize the left turn and the striaght forward way to the lake on that case.


At the end of the process, I found the most suitable was 4 epoches , I save the model as model.h5 and finally the vehicle is able to drive autonomously around the track without leaving the road.

**Phrase 2**
The vehicle was able to drive track 1 autonomously but it still failed to drive track 2. I believed it's due to the increase number of right curves, the Steep hill and the downhill, u-turn, shadow view, different brighness and completely different backgound on Track 2.

I first decide to add more augmented data such as random shadow and random brightness, and I also swap my model from LeNet to more complicated architecture nerual network, the Nvidia Network since the LeNet seems like it does not perfrom well after increase amount of data.

I save that model as model_nvidia.h5 adn I try on Track 1 and it succeed. However no matter how I tune the hyper parameters or increase more augmneted data, it still drive terribly on Track 2. I do some research and I also involved on a forum post and it seems like nobody has successfully drive on Track 2 with only Track 1 data, except for former Track 2(seems Track 2 changed on this term).


#### 2. Final Model Architecture

I create 2 model with differnt purposes: Lenet(model.h5) and Nvidia(model_nvidia.h5)

The main purpose of **Lenet(model.h5)** model is just to see if 2 lap of Track 1 data with extra flipping images, chopping images, recovery data from left and right camera and appropriate parameter tuning is enought for a vehicle to learn and drive autonomously on track 1 and it turns out it does! However, it perform really really terribly on Track 2, it can't even pass the first turn on the first few seconds.

**Here is a visualization of the architecture nvidian architecture**
![lenet]

Then I attempted to add extra more augmented data(random shadow, random birghtness) with more complicated **Nvidia(model_nvidia.h5)** architecture. It successfully pass Track 1 and it seems like it has slightly improvement on Track 2 depends on the screen resolutions.

**Here is a visualization of the architecture nvidian architecture**

![nvidia]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on Track 1 using center lane driving. 

since Track 1 has far more left turn than right turn, it create an imbalance of classes for the data, so I flip data randomly to create more balance data.

Further more, I used images generated from left and right camera to create more recovery data.

I also cropped the image to exclude part of the background view, so as to reduce noise for the training process. 
And I found one way to fix the failure on the left turn on Track 1 mentioned on above SECTION, **1. Solution Design Approach** is to crop slightly less so to include some background view on that special curve.


For nvidia model(model_nvidia.h5), I have extra augmented data for generalisation purpose. I augmented data with random shadow and random birghtness as I attempt to generalise the model to Track 2. But seems like it is impossible and one of the reason is that Track 2 has a huge difference with Track 1.

I finally randomly shuffled the data set and put 0.3 of the data into the Validation Set and 0.7 for the Training Set. 

Before the training start, I normalised the data to speed up the training process.

The following combinations I found working well for Track 1:
* epoch = 4
* cropping=((65,25), (0,0))
* measurment correction for images from left and right camera = +-0.20
* training and validation set split = 0.7:0.3

### Some Reflection
**Even if the vehicle is able to drive autonomously on Track one, there are still some issues I found**
* even when driving on straightway, the vehicle make some obvious steering angle instead of just ZERO steering angle
* fail to generalise to Track 2 with Track one and augmented data only

**I would have done the following next time for more improvements**
* collect more stable data(in other word I as a human will drive more stable when collecting training data)
* try to add some dropout to see if it could drive more stable on straightway instead of making obvious steering angle
* more plots for paremeter tuning so to make life easier to decide proper tuning value
* attempt more steering measurement correction value for images from left and right cameras 
