# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/7ee8d0d4-561e-4101-8615-66e0ab8ea8c8/concepts/a96fb396-2997-46e5-bed8-543a16e4f72e)
The above link directs you to my udacity workspace.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the Number of examples of each sign in the training set

![alt text][graph1.png]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided not to convert the images to grayscale because color may be irelevant cos a stop sign might be an upside-down red triangle, and knowing the colour of the triangle may help us predict the sign correctly.

I normalised the data. I found that normalising the data  gives the higher validation and training accuracy for a two-layer feedforward network. This is because it accelerates the convergence of the model to the solution (of accurate classificaion).

Here is an example of a traffic sign image before 

![alt text]['original.png']

after the normalisation

![alt text]['normalised.png']




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final architecture is a 3-layer Convolutional Neural Network.

It consists of one convolution layer (feature extraction) followed by two fully connected layers (ReLU activation) and a single fully connected linear classifier.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution layer1   	|stride = 3, VALID padding,  Filters =32 	|
| RELU					|	ReLu Activation	
| 2D Max pooling	   	| kernal size = 2,  outputs 5x5x32 Dropout :0.9

  Reshape covo1 output     input = 5x5x32 output = 800

 Input                    800
|Fully connected layer 1| output =512 WX+b Relu Activation function Dropout = 0.9

|Input                    512
|Fully connected layer 2| output = 128 WX+b Relu Activation function Dropout = 0.9        									|

 Input                    128
|Output layer			| output = 128  WX+b      									|
|						|												|
|						|												|
 
The networl uses the full color info(all 3 channels) and normalised data


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an optimiser called AdamOptimizser , Batch size is 100 , Traing epochs is 51 and the Learning data is .001
The Network parameters are as follows
Dropout(Convolutional layer) = .9
Dropout(Fully connected layer) = .9
Padding = VALID

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 0.9625 
* test set accuracy of 0.772209



If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

################ First attempt: building a minimum viable model and debugging
->I wanted to get a working model first. I started with a basic multilayer perceptron which I adapted from TensorFlow-Examples. I trained it for 15 epochs, which had an accuracy of 6% on the training and test sets. I then trained a two-layer convolutional neural network for 15 epochs which had an accuracy of 5-6% on the training and test sets.


* What were some problems with the initial architecture?

The accuracy was lower than I expected and the cost seemed high (of order 10^6 in the first epoch, 10^5 in the second and third and in the hundreds in the tenth epoch), so I adjusted parameters hoping to improve it before training for longer.

* How was the architecture adjusted and why was it adjusted?

The cost reduced significantly (to single digits by the second epoch as opposed to order 10^5) after I added a small positive bias to the initial weights and biases. Strangely, the accuracy did not increase, but remained at 5-6%. The cost did not decrease significantly over the next 10 epochs either.

I went on Slack to see what results people were getting to get a feel for how wrong I was. I saw that people often trained their networks for hundreds of epochs so I thought it would be good to train my network for e.g. 100 epochs.

I rewrote the multilayer perceptron in a Python Script and it worked fine, returning an accuracy of over 70% accuracy within 2 epochs.


Improvements to the model

I then added a convolution layer before the two fully connected layers and the output layer.This new model produced a validation accuracy of above 90% after 15 epochs (parameters not tuned), which was higher than that for the two-layer feedforward network. So I chose this model with a convolution layer.



Tuning Parameters

I altered the model code (replaced hard-coded numbers with variables) so I could tweak parameters easily.
I tested models with different values or settings for dropout for the fully connected layers,
dropout for the convolution layer,
padding (SAME vs VALID),
weight and bias initialisation
maxpool vs no max pool


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][japanese-sign.jpg] 
This is the Japanese stop sign,  it looks like a Yield sign. The network predicts this is a Roundabout Mandatory sign, which is completely different.

![alt text][german-sign.jpg] 

This is a no parking zone sign. The network predicts this is a 'Right-of-way at the next intersection' sign. They are not similar.
![alt text][two-way-sign.jpg] 
The network predicts this is a Go straight or left sign. They are similar in that there precisely two curved arrows in both signs.

![alt text][speed-limit-stop.jpg] 
The network predicts this is a roundabout mandatory sign (40). This is wrong- it should be 20km/h speed limit . The network may have been confused by the many curves that make up the sign.


![alt text][shark-sign.jpg]

The network predicts this is a roundabout mandatory sign. This is wrong, but then there is no correct class within the 43 for this sign.
    
It is unclear why this sign should be the roundabout mandatory sign of all signs.
        
There are not many curved arrows - the black portion of the sign is small and is close to a short horizontal line segment in the middle of the sign.
The diamond-shaped sign could have indicated Priority Road (12).



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Roundabout Mandatory sign 									| 
| No parking     			| Right-of-way at the next intersection 										|
| Two way sign					| Go straight or left sign											|
| Speed limit stop	      		| roundabout mandatory sign (40)					 				|
| Shark sign			| roundabout mandatory sign      							|

No, it does not perform equally well on captured images(new traffic signs). It has a performance of 0% accuracy on captured images as opposed to 79% on the test set.
        The images not included in the dataset are not exactly the same road signs so there is additional difficulty because the model needs to generalise well to classify these new signs correctly. The
        Some road signs such as the shark sign may not even be included in the 43 categories.
        The images are also processed (e.g. cropped) differently.
        
It seems that the model is classifying 'unknown signs' as Roundabout Mandatory signs.

Reference for images of correct German signage: http://www.gettingaroundgermany.info/zeichen.shtml


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located from in[231] : to in[246] cell of the Ipython notebook.

The model is certain of all of its predictions even though some are wrong.

The model also predicts different outcomes confidently for the two times I ran the predictions on each sign.

These are both strange outcomes.

For the first image, The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .75         			| Keep right   									| 
| .25     				| Go straight or right 										|
| .00					| Roundabout mandatory										|
| .00	      			| Speed limit (30km/h)					 				|
| .00				    | End of no passing      							|


For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Children crossing   									| 
| .05     				| Speed limit (30km/h) 										|
| .00					| No passing										|
| .00	      			| Speed limit (50km/h)					 				|
| .00				    | End of no passing      							|

for the third image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Roundabout mandatory   									| 
| .00     				| Go straight or left 										|
| .00					| Turn right ahead										|
| .00	      			| Turn left ahead					 				|
| .00				    | Ahead only      							|

for the fourth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)   									| 
| .00     				| Roundabout mandatory 										|
| .00					| Speed limit (50km/h)										|
| .00	      			| Keep right					 				|
| .00				    | Children crossing      							|

for the fifth  image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead only   									| 
| .00     				| Roundabout mandatory 										|
| .00					| Turn left ahead										|
| .00	      			| Go straight or left					 				|
| .00				    | Go straight or right     							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


