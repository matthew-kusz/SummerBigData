# Import the necessary packages
import numpy as np				 	 # Allow for easier use of arrays and linear algebra
import pandas as pd                         	 	 # For reading in and writing files
import matplotlib.image as mpimg           	 	 # Reading images to numpy arrays
import cv2
from keras.utils import np_utils			 # Used to set up one-hot scheme
from scipy.misc import imresize                  	 # For resizing the images
from sklearn.decomposition import PCA           	 # Preprocessing
from sklearn.preprocessing import LabelEncoder  	 # Preprocessing

import matplotlib.pyplot as plt

####### Definitions #######
def grab_images(tr_ids, te_ids, tot_img):
	'''
	Reads in image files to save to a list, then separates the pictures between training and testing ids

	Parameters:
	tr_ids - 1D array of training image ids
	te_ids - 1D array of testing image ids

	Returns:
	images_list - full list of images
	train_list - list only consisting of training images
	test_list - list only consisting of testing images
	'''	

	# Full set
	matrix = np.zeros((2, tot_img))
	images_list = []
	for i in range(tot_img):
		img = mpimg.imread('data_provided/unzip_images/images/' + str(i + 1) + '.jpg')
		matrix[:,i] = np.shape(img)
		images_list.append(img)

	# We will want to learn features on the images that belong to our training set	
	train_list = []
	for i, img in enumerate(tr_ids):
		train_list.append(images_list[img - 1])
	
	# We might want to look at some test images too	
	test_list = []
	for i, img in enumerate(te_ids):
		test_list.append(images_list[img - 1])

	return images_list, train_list, test_list

def resize_img(img, max_dim):
	'''
	Resize the image so the maximum side is of size max_dim
	and the other side is scaled accordingly
	
	Parameters:
	img - image being resized
	max_dim - maximum dimension of image that was requested

	Returns a new image of the right size
	'''

    	# Get the axis with the larger dimension
    	max_ax = max((0, 1), key=lambda i: img.shape[i])

    	# Scale both axes so the image's largest dimension is max_dim
   	scale = max_dim / float(img.shape[max_ax])

    	return np.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)))

def reshape_img(images, max_dim, center = True):
	'''
	Reshapes images to the requested dimensions

	Parameters:
	images - list of the images that are being resized
	max_dim - maximum dimension of images that was requested
	center - whether or not the picture will be placed in the center when modified

	Returns a 2D array of the modified images
	'''

	# Initialize the array that will hold the modified images
	modified = np.zeros((len(images), max_dim, max_dim, 1))

	# Reshape each image to the requested dimensions
	for i in range(len(images)):
		temp = resize_img(images[i], max_dim = max_dim)
		x = imresize(images[i], (temp.shape[0], temp.shape[1]), interp = 'nearest').reshape(temp.shape[0], 
			temp.shape[1], 1)

		length = x.shape[0]
		width = x.shape[1]

		# Place in center of True
		if center:
			h1 = int((max_dim - length) / 2)
           	 	h2 = h1 + length
           		w1 = int((max_dim - width) / 2)
          		w2 = w1 + width
       		else:
           		h1, w1 = 0, 0
          		h2, w2 = (length, width)
	
		modified[i, h1:h2, w1:w2, 0:1] = x

	return  np.around(modified / 255.0)

def engineered_features(train, test, tr_list, te_list):
	'''
	Create more features that can be obtained from characteristics of the images

	Parameters:
	te_list - list of the training images
	tr_list - list of the testing images
	train - 2D array of pre-extracted features for the training set
	test - 2D array of pre-extracted features for the testing set

	Return:
	test_mod - 2D array of pre-extracted features for the training set with engineered features
	train_mod - 2D array of pre-extracted features for the test set with engineered features
	'''

	print 'Grabbing more features...'
	# Initialize each array
	tr_width = np.zeros((len(tr_list), 1)) 
	tr_height = np.zeros((len(tr_list), 1))
	tr_asp_ratio = np.zeros((len(tr_list), 1))
	tr_square = np.zeros((len(tr_list), 1))
	tr_mean = np.zeros((len(tr_list), 1))
	tr_horiz = np.zeros((len(tr_list), 1))
	te_width = np.zeros((len(te_list), 1))
	te_height = np.zeros((len(te_list), 1))
	te_asp_ratio = np.zeros((len(te_list), 1))
	te_square = np.zeros((len(te_list), 1))
	te_mean = np.zeros((len(te_list), 1))
	te_horiz = np.zeros((len(te_list), 1))

	# Calculate the features of the training images
	for i in range(len(tr_list)):
		tr_width[i] = tr_list[i].shape[1]
		tr_height[i] = tr_list[i].shape[0]
		tr_asp_ratio[i] = tr_list[i].shape[1] / tr_list[i].shape[0]
		tr_square[i] = tr_list[i].shape[1] * tr_list[i].shape[0]
		tr_mean[i] = np.mean(tr_list[i])
		if (tr_width[i] < tr_height[i]):
			tr_horiz[i] = 1
		else:
			tr_horiz[i] = 0

	# Calculate the features of the test images
	for i in range(len(te_list)):
		te_width[i] = te_list[i].shape[1]
		te_height[i] = te_list[i].shape[0]
		te_asp_ratio[i] = te_list[i].shape[1] / te_list[i].shape[0]
		te_square[i] = te_list[i].shape[1] * te_list[i].shape[0]
		te_mean[i] = np.mean(te_list[i])
		if (te_width[i] < te_height[i]):
			te_horiz[i] = 1
		else:
			te_horiz[i] = 0

	# Attach these features to the pre-extracted ones
	train_mod = np.concatenate((train, tr_width, tr_height, tr_asp_ratio, tr_square, tr_mean, tr_horiz),
		axis = 1)
	test_mod = np.concatenate((test, te_width, te_height, te_asp_ratio, te_square, te_mean, te_horiz),
		axis = 1)

	print 'Finished.'
	return train_mod, test_mod

def more_features(train, test, tr_arr, te_arr):

	print tr_arr[0].max()
	ret, thresh = cv2.threshold(tr_arr[0], 127, 255, 1)
	_, contours, _ = cv2.findContours(thresh, 2, 1)
	plt.imshow(thresh, cmap = 'binary', interpolation = 'none')
	plt.show()
	print contours[0].max()
	cnt = contours[0]
	M = cv2.moments(cnt)

	hull = cv2.convexHull(cnt)
	

	hull = cv2.convexHull(cnt,returnPoints = False)
	defects = cv2.convexityDefects(cnt,hull)

	for i in range(defects.shape[0]):
    		s,e,f,d = defects[i,0]
    		start = tuple(cnt[s][0])
    		end = tuple(cnt[e][0])
    		far = tuple(cnt[f][0])
    		cv2.line(tr_arr[0],start,end,[0,255,0],2)
    		cv2.circle(tr_arr[0],far,5,[0,0,255],-1)

	plt.imshow(tr_arr[0], cmap = 'gray', interpolation = 'bicubic')
   	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    	plt.show()

	train_moments = np.zeros((len(tr_arr), 1))
	for i in range(len(tr_arr)):
		stop
	return

def apply_PCA(train, test, tr_mod_list, te_mod_list, max_dim):
	'''
	Use PCA to create lower dimensional images that can be used with the pre-extracted features

	Parameters:
	te_mod_list - list of the resized training images
	tr_mod_list - list of the resized testing images
	train - 2D array of pre-extracted features for the training set
	test - 2D array of pre-extracted features for the testing set

	Return:
	train - 2D array of pre-extracted features for the training set with the flattened PCA features
	test - 2D array of pre-extracted features for the testing set with the flattened PCA features
	'''

	print 'Applying PCA...'

	tr_flat = np.zeros((len(tr_mod_list), max_dim * max_dim))
	for i in range(len(tr_mod_list)):
		tr_flat[i] = tr_mod_list[i].ravel()

	te_flat = np.zeros((len(te_mod_list), max_dim * max_dim))
	for i in range(len(te_mod_list)):
		te_flat[i] = te_mod_list[i].ravel()
	
	pca = PCA(n_components = 30)
	pca.fit(tr_flat)
	tr_flat_pca = pca.transform(tr_flat)
	te_flat_pca = pca.transform(te_flat)

	'''
	print pca.explained_variance_ratio_
	print np.sum(pca.explained_variance_ratio_)
	approximation = pca.inverse_transform(tr_flat_pca)
	
	# Relationship of information retained vs. number of prinicpal components
	mark = [15, 40, 70, 300]
	plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='b', markevery = mark)
	plt.title('Information Retained vs. Principal Components')
	plt.ylabel('Fraction of Information Retained')
	plt.xlabel('Number of Principal Components')
	plt.show()

	plt.figure(figsize=(8,4))
	# Original Image
	plt.subplot(1, 2, 1)
	plt.imshow(tr_mod_list[0].reshape(50,50),
              	cmap = plt.cm.gray, interpolation='nearest',
              	clim=(0, 1))
	ax = plt.gca()
	ax.axes.set_xticklabels([])
	ax.axes.get_yaxis().set_visible(False)
	plt.xlabel('2500 components', fontsize = 14)
	plt.title('Original Image', fontsize = 20)

	# 154 principal components
	plt.subplot(1, 2, 2)
	plt.imshow(approximation[0].reshape(50, 50),
              	cmap = plt.cm.gray, interpolation='nearest',
              	clim=(0, 1))
	ax = plt.gca()
	ax.axes.set_xticklabels([])
	ax.axes.get_yaxis().set_visible(False)
	plt.xlabel('300 components', fontsize = 14)
	plt.title('PCA Image', fontsize = 20)
	plt.show()
	'''
	
	train = np.concatenate((train, tr_flat_pca), axis = 1)
	test = np.concatenate((test, te_flat_pca), axis = 1)
	
	print 'Finished.'
	return train, test

def data():
	'''
	Load in and setup the data 

	Returns:
	train_list - list of training images
	test_list - list of testing images
	train_ids - 1D array of the training set leaves ids
	test_ids - 1D array of the testing set leaves ids
	train - 2D array pre-extracted features of the training set leaves
	test- 2D array pre-extracted features of the testing set leaves
	y - 1D array that holds a label for each species ranging from 0 to n - 1
	y_train - 2D array set up with the one hot scheme
	classes - 1D array of the name of each leaf species
	'''	

	# Set up our training data
	train = pd.read_csv('data_provided/train.csv')

	# Extract the species of each leaf
	y_raw = train.pop('species')

	# Label each species from 0 - n-1
	le = LabelEncoder()
	# fit() calculates the mean and std, transform() centers and scales data
	y = le.fit(y_raw).transform(y_raw)
			
	# Grab the classes (will be used to set up our submission file)
	classes = le.classes_
	# Setting up one-hot scheme
	y_train = np_utils.to_categorical(y)

	# Extract the id of each leaf
	train_ids = train.pop('id')

	# Set up our testing data
	test = pd.read_csv('data_provided/test.csv')

	# Extract the id of each leaf
	test_ids = test.pop('id')

	# Find our how many images we have
	total_img = len(train_ids) + len(test_ids)

	# Load up all of the images
	print 'Loading images...'
	img_list, train_list, test_list = grab_images(train_ids, test_ids, total_img)
	print 'Finished.'

	return train_list, test_list, train_ids, test_ids, train, test, y, y_train, classes