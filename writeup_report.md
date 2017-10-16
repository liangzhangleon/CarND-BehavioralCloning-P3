---

**Behavioral Cloning Project**
**Liang Zhang**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
 Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

#### 1. My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 recording the car in autonomous mode
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is derived from the model of the autonomous vechicle team in Nvidia. 

The input data is normalized in the model using a Keras lambda layer (model.py line 45), which is followed by a cropping layer. 

The model contains five convolutional layers with 5x5 or 3x3 filter sizes and depths between 24 and 64 (model.py lines 47-51). After each convolutional layer,  a RELU layer is included to introduce nonlinearity (model.py lines 47-51). 

Four fully connected layers are followed with a single output (model.py lines 53-54, 56-57). 

#### 2. Attempts to reduce overfitting in the model

Afte the second fully connected layers, a dropout layer is included (model.py lines 55).

The model was trained and validated on different data sets by using data splitting (model.py line 59). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 59).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I only used the center lane driving data provided as the sample data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 45-57) :


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first checked the sample data and found that the sample data shows good center driving behavior. Hence, I used this data set. Here is an example image of center lane driving:

![alt text](center_2016_12_01_13_30_48_287.jpg)

To augment the data sat, I also flipped images and angles thinking that this would help. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

In the sample data, I had 8036 number of data points. I augumented the data set by flipping the image. I then preprocessed this data by normalization and cropping the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. The validation set helped determine if the model was over or under fitting. The number of epochs I used was 5 by checking the validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
