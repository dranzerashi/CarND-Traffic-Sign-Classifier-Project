# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project were the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./writeup_images/augumentation.png "Rotation and Translation"
[image4]: ./test_images/testimg1.jpg "Traffic Sign 1"
[image5]: ./test_images/testimg2.jpg "Traffic Sign 2"
[image6]: ./test_images/testimg3.jpg "Traffic Sign 3"
[image7]: ./test_images/testimg4.jpg "Traffic Sign 4"
[image8]: ./test_images/testimg5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dranzerashi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Here is a link to my [HTML Extract](https://github.com/dranzerashi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the initial data is split across categories.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Initially I decided to convert the images to HSV and stack them on top of RGB turning it into 6 channel image because HSV seperates the color and intensity information into different channels similar to grayscaling without losing the color information. This gave me good results with LenetV2 architecture that I initially made. However Normalization after this step caused a huge leap in memory requirements(as uint8 to float32 times 6 channels) causing system crashes. Hence I tried seperately with HSV 3 channel and RGB and found that there was not much difference in performance between them. So I settled with a RGB 3 channel with Normalization. This caused a larger amount of processing time per batch due to float operations but made the loss curve much smoother.

I decided to generate additional data because I found that my model performed badly when classifying the images with less amount of data/class. For eg initially the Mandatory Round About sign was classified as a Stay Right sign. Augumenting these classes with fake data solved this problem.  

To add more data to the the data set, I used the following techniques:
1. Translation along the X,Y axis randomly by upto 10%,
2. Rotation of the image in either direction randomly by upto 8degrees
3. A combination of the two above.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is that most images that are having lower than 1500 samples have been augumented by rotation and translation to much more samples


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB Normalized image   							| 
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 32x32x3 	|
| RELU					|									
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| flatten               | outputs 400 	                                |
| Fully connected 400x120| bias 120, outputs 120						|
| RELU					|												|
| Fully connected 120x84| bias 84, outputs 84							|
| RELU					|												|
| Fully connected 84x43 | bias 43, outputs 43							|
|						|												|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with Softmax Cross Entropy with logits. I used 0.001 as learning rate with 50 epochs and a batch size of 128.
I found that the learning proceeded at a smooth and steady rate with a dropout probability of 50%.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.974 
* test set accuracy of 0.953

The initial architecture was a Standard LeNet architecture adapted for 6 channels (stack of RGB and HSV) this is found in ```LeNetV2()```, This architecture gave me a validation accuracy of about 93% with augumentation at its best result. However When I applied Normalization this immediately caused system crashes due to increased memory requirements of float32 values. Hence I settled for RGB as HSV individually did not give me any better results.
Hence I went back to a standard LeNet architecture with Normalization. And without augumentation I was able to achive around 95% to 96% validation accuracy. But the training loss and Validation loss were extremely jittery and Validation loss was worse for classes without augumentation. I added more augumented images for these classes. I also added Dropout to smoothen the learning. I also Modified the Lenet architecture to use a 1x1 convolution followed by ReLU and no pooling to give it more depth this avoided overfitting on the training set. This architecture gave me results that at times gave best results of upto 97.7% on the validation set. The final result was 95.3% in the test set.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image was difficult to classify because the training set had very few sets of this class. Initially it was calssified as Stay Right sign due to this. Augumenting the data set however solved the problem

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Slippery road			| Slippery road									|
| Turn right ahead		| Turn right ahead								|
| Speed limit (30km/h)	| Speed limit (30km/h)			 				|
| Roundabout mandatory	| Roundabout mandatory 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 34th cell of the Ipython notebook.

For the first image, the model is sure that this is a stop sign (probability of ~1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop sign   									| 
| 7.213995e-11			| No vehicles									|
| 6.341166e-12      	| Speed limit (30km/h)							|
| 5.399329e-13 			| Yield     					 				|
| 1.245141e-13		    | Speed limit (60km/h) 							|


For the second image, the model is sure that this is a Slippery road sign (probability of ~1.0), and the image does contain a Slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Slippery road									| 
| 4.306018e-09			| Dangerous curve to the right              	|
| 3.6228173e-12      	| Dangerous curve to the left               	|
| 1.0730475e-16			| Right-of-way at the next intersection			|
| 1.8337623e-21		    | Children crossing 							|

For the third image, the model is sure that this is a Turn right ahead sign (probability of ~1.0), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn right ahead								| 
| 1.2732035e-13			| Ahead only                                  	|
| 6.56296e-14         	| Turn left ahead                              	|
| 1.3182148e-17			| Keep left                         			|
| 9.014684e-18		    | Go straight or left  							|

For the fourth image, the model is sure that this is a Speed limit (30km/h) ahead sign (probability of 0.999), and the image does contain a Speed limit (30km/h) ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99986684   			| Speed limit (30km/h)							| 
| 0.00013314499			| Speed limit (20km/h)                         	|
| 4.788935e-10         	| Speed limit (50km/h)                         	|
| 4.647488e-13			| Speed limit (70km/h)                 			|
| 1.1079947e-13		    | Speed limit (60km/h) 							|

For the final image, the model is sure that this is a Roundabout mandatory sign (probability of 0.99), and the image does contain a Roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9907857   			| Roundabout mandatory							| 
| 0.0091924155			| Go straight or left                         	|
| 1.0634703e-05        	| Ahead only                                 	|
| 7.907677e-06			| Turn left ahead                    			|
| 3.1629243e-06		    | Turn right ahead   							|

This final image initially was completely misclassified initially. However Image augumentation gave it a very strong accuracy of 0.99

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I have not tried this as of now due to time limitations. I will be sure to update this section later on.

### Additional Remarks
1. I have added a few initial code that automatically download the Compressed Dataset and extracts it to the respective folder ```\traindata```. These are found on line 2 and 3 of the python notebook.
2. The 5 test images are saved in the ```\test_images``` folder.
3. I have additionally saved and reported the test performance of the model that was found to be the one having the highest training and validation accuracy among all the epochs as ```./best-lenet```. The result is on line 15 of the notebook.
4. As opencv is not by default installed on aws carnd image the following command needs to be run to use it:```apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python```
5. I have printed the top 5 probability as a table with sign names instead of indices by using the reading the signnames.csv
