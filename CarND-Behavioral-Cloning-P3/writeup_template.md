# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive1.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
The code is functional, it can be tested by executing:
python model.py

Using the Udacity provided simulator and my drive1.py file, the car can be driven autonomously around the track by executing 
```sh
python drive1.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The rubric suggested using a Python generator, and one is included in the code. That said, in my local machine I found that not using a generator provided me with better results given the memory requirements of the model and the size of the input data. I ran my model in a GPU-accelerated machine with 8 GBs of video memory, which could accommodate and execute the model in training with batches of 512 images per cycle.

The use of Python generators allows us to not be memory constrained, but its tradeoff is time. When I resorted to using a Python generator, each epoch took around 85 seconds to complete, as the processing of the images required the generation of the entire set on each epoch, and used CPU cycles that were slower than the GPU alone could do with a predefined data set. I used the fit_generator function to train the model using a Python generator.
From model.py lines 136-165 :



### Model Architecture and Training Strategy

This project challenged my understanding of neural networks in that, so far, I’ve only been doing categorization problems, such as the traffic sign categorization done in Traffic sign classifier. When I started behavioral cloning it wasn’t clear to me how I was going to build a model to do what appeared to be a regression model. The Keras documentation for sequential model compilation[https://keras.io/getting-started/sequential-model-guide/#compilation] was useful in telling me two things:

>Loss calculation for regression problems differs from that of categorization problems, and
>Mean squared error is what I was looking for.

In retrospective, this should have been intuitive from the material we’ve seen. Given that I could recognize the need for mean squared error, I went and created a simple sequential model with this architecture:

1>Convolution (32 filters, 3x3 kernel, 1x1 stride, valid mode)
2>Activation (ReLu)
3>Max pooling (2x2 pool size)
4>Dropout
5>Convolution (64 filters, 2x2 kernel, 1x1 stride, valid mode)
6>Activation (ReLu)
7>Max pooling (2x2 pool size)
8>Dropout
9>Flatten
10>Fully connected (128 output)
11>Activation (ReLu)
12>Dropout
13>Fully connected (1)
14>Activation (Relu)
15>Compile (Adam optimizer, MSE loss)

I added an activation layer in step 14, before compilation, meaning that whatever value the fully connected layer was giving me in step 13 was going to get “activated” in the way we usually produce categorization problems, yielding a zero when the value from the previous layer was negative, and a one otherwise.

My first run was problematic, of course, but after going back to my notes I realized what I had made. I based this model from what we had done in the introduction to Keras, which was a categorization problem. After it became clear that I was looking for a result set in the range [-1, 1], I realized that the activation layer was damaging my results. After removing it, the model started to produce values that seemed plausible.

I recurred to the NVidia whitepaper titled End-to-End Learning for Self-Driving Cars [http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf] which gave me a much better architecture:

Convolution (24 filters, 5x5 kernel, 2x2 stride, valid mode)
Activation (ReLu)
Dropout
Convolution (36 filters, 5x5 kernel, 2x2 stride, same mode)
Activation (ReLu)
Dropout
Convolution (48 filters, 5x5 kernel, 2x2 stride, same mode)
Activation (ReLu)
Dropout
Convolution (64 filters, 3x3 kernel, 1x1 stride, same mode)
Activation (ReLu)
Dropout
Convolution (64 filters, 3x3 kernel, 1x1 stride, same mode)
Activation (ReLu)
Dropout
Flatten
Fully connected (1164 output)
Activation (ReLu)
Dropout
Fully connected (100 output)
Activation (ReLu)
Dropout
Fully connected (50 output)
Activation (ReLu)
Dropout
Fully connected (1)
Activation (Relu)
Compile (Adam optimizer, MSE loss)

once you run the model , you can see the summary  of the model.

####  Attempts to reduce overfitting in the model

The architecture does contain dropout layers, which by default have been configured to be 50% during model training.

Additionally, the data used to train the model has been augmented to make the neural network both less reliant on the initial data set and more flexible so it understands variations in the input that might not have been captured initially.
#### . Model parameter tuning

I initially experimented with different values for batch size and number of epochs. I was using the default Adam optimizer that Keras provides. After reading some documentation I realized that the Adam optimizer can be further fine-tuned with regards to its learning rate, as well as a few other parameters. I noticed a big improvement when I chose to make the learning rate smaller than the default setting was, while also running a larger number of epochs.

I settled on the following values:

Number of epochs: 1000 (model.py line 268)
Batch size: 512 (model.py line 268)
Learning rate for the Adam function: 0.00005 (model.py line 167)
Dropout rate during training: 0.5 (model.py line 167)

 I loaded the images and did the following transformations:

>Trim the image so only the road section is fed into the model
>Resize to 64 pixels wide by 16 pixels tall.

The trimming was done by keeping the entire width of the image, but only keeping the window of pixes from top = 60 to bottom = 140. These values were determined "by hand", simply by opening a sample set of images in an editor and figuring out that this is where most of them would contain the road information. 

The resizing of the image to 64x16 was intentionally made to keep the trimmed image form factor while reducing the memory requirements. The original image after being trimmed was still 320x120 and I wanted to approximate something closer to 32x32 initially, just to keep the memory pressure down.

Following that, I wanted to make use of the left and right camera data, not only the center camera data. Given that these cameras are offset from the center I modified the steering angle associated to them as indicated in the NVidia whitepaper. From the whitepaper:[http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf]

Images for two speciﬁc off-center shifts can be obtained from the left and the right camera. Additional shifts between the cameras and all rotations are simulated by viewpoint transformation of the image from the nearest camera. Precise viewpoint transformation requires 3D scene knowledge which we don’t have. We therefore approximate the transformation by assuming all points below the horizon are on ﬂat ground and all points above the horizon are inﬁnitely far away. This works ﬁne for ﬂat terrain but it introduces distortions for objects that stick above the ground, such as cars, poles, trees, and buildings. Fortunately these distortions don’t pose a big problem for network training. The steering label for transformed images is adjusted to one that would steer the vehicle back to the desired location and orientation in two seconds.

I didn’t do any precise transformations the way they claimed they did, I only took the right and left camera images and adjusted the steering by 0.15. The left camera makes the road look like it’s curving to the right, so I added 0.15 to the steering value on the left camera. The right camera makes the road look like it’s curving to the left, so I subtracted 0.15. I then capped the values at the [-1,1] range originally provided. See model.py lines 118 to 125 for the implementation.

I chose 0.15 as the steering value with no basis on what number to use. I’m not sure how optimal or suboptimal this value is, given that the result was satisfactory and I didn’t adjust it much. I did play with 0.1 instead, but I reverted to 0.15 and the car was able to drive with this as expected.

After that, I ran three more transformations on the data:

Leaving only 10% of the center data
Balancing left and right data by mirroring the entire dataset
Normalizing the image data
I left only 10% of the center data as it originally was disproportionally large compared to any other value when visualized in a histogram. I did this so that the model wouldn’t overfit too much to the center.

The Data preprocessing and normalization done in balance(model.py line 37-66) and normalize_minmax functions(model.py line 68-78)

The balancing of the left and right data was simple. I took all the images and flipped them using the cv2 library, and I negated its associated steering value. The resulting set was merged with the original one, thus giving me twice the amount of data I started with.

The data normalization ensuring all the pixel values would be [-0.5 to 0.5]

after traing the model, i ran the car in the autonomous mode and collected the image frames and saved in run1 folder 
using the video.py and executing the below command 
python video.py run1
I created a video based on images found in the run1 directory.The name of the video will be the name of the directory followed by '.mp4', so, in this case the video will be run1.mp4.








