# Import the necessary packages
import numpy as np				 	 # Allow for easier use of arrays and linear algebra
import pandas as pd                         	 	 # For reading in and writing files
import matplotlib.image as mpimg           	 	 # Reading images to numpy arrays
import cv2						 # For grabbing more features
from keras.utils import np_utils			 # Used to set up one-hot scheme
from scipy.misc import imresize                  	 # For resizing the images
from sklearn.decomposition import PCA           	 # Preprocessing
from sklearn.preprocessing import LabelEncoder  	 # Preprocessing

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import visualize
import matplotlib.pyplot as plt
from skimage.measure import regionprops
####### Definitions #######
def grab_images(tr_ids, te_ids, tot_img):
	'''
	Reads in image files to save to a list, then separates the pictures between training and testing ids

	Parameters:
	tr_ids - 1D array of training image ids
	te_ids - 1D array of testing image ids
	tot_img - number of images in total

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
		tr_asp_ratio[i] = tr_list[i].shape[1] / float(tr_list[i].shape[0])
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
		te_asp_ratio[i] = te_list[i].shape[1] / float(te_list[i].shape[0])
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

def more_features(train, test, tr_list, te_list, y, classes):
	'''
	Grab for features to learn from using openCV

	Parameters:
	tr_list - list of the training images
	te_list - list of the testing images
	train - 2D array of pre-extracted features for the training set
	test - 2D array of pre-extracted features for the testing set

	Return:
	test_mod - 2D array of pre-extracted features for the training set with additional features
	train_mod - 2D array of pre-extracted features for the test set with additional features
	'''

	tr_area = np.zeros((len(tr_list), 1)) 
	tr_per = np.zeros((len(tr_list), 1))
	tr_hull = np.zeros((len(tr_list), 1))
	tr_cx = np.zeros((len(tr_list), 1))
	tr_cy = np.zeros((len(tr_list), 1))
	tr_MA = np.zeros((len(tr_list), 1))
	tr_ma = np.zeros((len(tr_list), 1))
	tr_angle = np.zeros((len(tr_list), 1))
	tr_mom = np.zeros((len(tr_list), 24))
	te_area = np.zeros((len(te_list), 1)) 
	te_per = np.zeros((len(te_list), 1))
	te_hull = np.zeros((len(te_list), 1))
	te_cx = np.zeros((len(te_list), 1))
	te_cy = np.zeros((len(te_list), 1))
	te_MA = np.zeros((len(te_list), 1))
	te_ma = np.zeros((len(te_list), 1))
	te_angle = np.zeros((len(te_list), 1))
	te_mom = np.zeros((len(te_list), 24))
	
	tr_mom2 = np.zeros((len(tr_list), 24))
	te_mom2 = np.zeros((len(te_list), 24))
	count = 0
	temp = []
	temp2 = []
	temp3 = []
	arr1 = np.zeros(99)
	arr2 = np.zeros(99)
	# Find the moments of each leaf in the taining set
	for i in range (len(tr_list)):
		ret, thresh = cv2.threshold(tr_list[i],127,255,0)	
	
		im , contours, _ = cv2.findContours(thresh, 1, 2)
		'''
		FOR DEBUGGING
		for j in range(len(contours)):
			if j + 1 == len(contours):
				print len(contours)
				if len(contours) > 5:
					print 'New Leaf'
					for k in range(len(contours)):
						print '# of contour points: ', len(contours[k])

				     		for h in range(len(contours[k])):
				     			print 'Point(x,y)=', contours[k][h]

				     		print 'Area: ', cv2.contourArea(contours[k])
						print 'perimeter', cv2.arcLength(contours[k],True)

					plt.imshow(im, cmap = 'binary')
					plt.show()
		'''
		
		cnt = contours[0]
		M = cv2.moments(cnt, binaryImage = True)
		k = 0
		for key in M:
			tr_mom[i][k] = M[key] / 255.0
			k += 1
		
		'''
		tr_per[i] += cv2.arcLength(cnt,True) / 255.0
		
		if M['m00'] > 0:
			temp.append(train[i])
			temp2.append(tr_mom[i])
			temp3.append(y[i])

		if M['m00'] == 0:
			count += 1
			arr1[y[i]] += 1 
			print M['m00']
			print classes[y[i]]
		'''
		'''
		hull = cv2.convexHull(cnt,returnPoints = False)
		defects = cv2.convexityDefects(cnt,hull)
		all_defects = 0	
		for k in range(len(defects)):
			all_defects += defects[k][0][3]
		all_defects /= len(defects)
		tr_hull[i] = all_defects / 255.0
		'''
		'''
		(x, y),(MA, ma), angle = cv2.fitEllipse(cnt)
		tr_MA[i] = MA / 255.0
		tr_ma[i] = ma / 255.0
		tr_angle[i] = angle		
		'''
		'''
		count += 1
 		for j in range(len(contours)):
			cnt = contours[j]
			M = cv2.moments(cnt, binaryImage = True)
			if M['m00'] > 5000:
				k = 0
				for key in M:
					tr_mom[i][k] = M[key] / 255.0
					k+= 1
		if count % 10 == 0:
			print 'Here!'
			k = 0
			for key in M:
				tr_mom[i][k] = 0
				k+= 1
		'''
		'''
			#tr_area[i] += cv2.contourArea(cnt) / 255.0
			#tr_per[i] += cv2.arcLength(cnt,True) / 255.0
			
			if cv2.contourArea(cnt) > 5000:
				M = cv2.moments(cnt, binaryImage = True)

				tr_cx[i] =  int(M['m10'] / M['m00']) / 255.0
				tr_cy[i] =  int(M['m01'] / M['m00']) / 255.0
			
				hull = cv2.convexHull(cnt,returnPoints = False)
				defects = cv2.convexityDefects(cnt,hull)
				all_defects = 0	
				for k in range(len(defects)):
					all_defects += defects[k][0][3]
				all_defects /= len(defects)
				tr_hull[i] = all_defects / 255.0

				(x, y),(MA, ma), angle = cv2.fitEllipse(cnt)
				tr_MA[i] = MA / 255.0
				tr_ma[i] = ma / 255.0
				tr_angle[i] = angle
		'''
		
		'''
		if M['m10'] != 0:
			#print regionprops(tr_list[i])
			tr_cx[i] =  int(M['m10'] / M['m00'])
			tr_cy[i] =  int(M['m01'] / M['m00'])
		else:
			print 'Printing...'
			#print contours[1]
			print 'Printing...'
			#print contours
			print i
			plt.imshow(tr_list[i], cmap = 'binary')
			plt.show()
			print cnt
			count += 1 

		hull = cv2.convexHull(cnt,returnPoints = False)
		defects = cv2.convexityDefects(cnt,hull)
		all_defects = 0
		if hull.all() != 0:
			for j in range(len(defects)):
				all_defects += defects[j][0][3]
			all_defects /= len(defects)
			tr_hull[i] /= all_defects
		else:	
			tr_hull[i] = all_defects
		'''
	print arr1
	print '\n'
	print count 
	print '\n'

	count = 0
	# Find the moments of each leaf in the test set
	for i in range (len(te_list)):
		ret, thresh = cv2.threshold(te_list[i],127,255,0)	
	
		im , contours, _ = cv2.findContours(thresh, 1, 2)
		
		cnt = contours[0]
		M = cv2.moments(cnt, binaryImage = True)
		
		k = 0
		for key in M:
			te_mom[i][k] = M[key] / 255.0
			k+= 1
		
		'''
		te_per[i] += cv2.arcLength(cnt,True) / 255.0
		
		if M['m00'] == 0:
			count += 1
			arr2[y[i]] += 1
			print classes[y[i]]
		'''
		'''
		hull = cv2.convexHull(cnt,returnPoints = False)
		defects = cv2.convexityDefects(cnt,hull)
		all_defects = 0	
		for k in range(len(defects)):
			all_defects += defects[k][0][3]
		all_defects /= len(defects)
		te_hull[i] = all_defects / 255.0
		'''
		'''
		(x, y),(MA, ma), angle = cv2.fitEllipse(cnt)
		tr_MA[i] = MA / 255.0
		tr_ma[i] = ma / 255.0
		tr_angle[i] = angle
		'''
		'''
		count += 1
 		for j in range(len(contours)):
			cnt = contours[j]
			M = cv2.moments(cnt, binaryImage = True)
			if M['m00'] > 5000:
				k = 0
				for key in M:
					te_mom[i][k] = M[key] / 255.0
					k+= 1

		if count % 10 == 0:
			k = 0
			for key in M:
				te_mom[i][k] = 0
				k+= 1
		'''
		'''

			te_area[i] += cv2.contourArea(cnt) / 255.0
			te_per[i] += cv2.arcLength(cnt,True) / 255.0

			if cv2.contourArea(cnt) > 5000:
				M = cv2.moments(cnt, binaryImage = True)
				te_cx[i] =  int(M['m10'] / M['m00']) / 255.0
				te_cy[i] =  int(M['m01'] / M['m00']) / 255.0
			
				hull = cv2.convexHull(cnt,returnPoints = False)
				defects = cv2.convexityDefects(cnt,hull)
				all_defects = 0				
				for k in range(len(defects)):
					all_defects += defects[k][0][3]
				all_defects /= len(defects)
				te_hull[i] = all_defects / 255.0

				(x, y),(MA, ma), angle = cv2.fitEllipse(cnt)
				te_MA[i] = MA / 255.0
				te_ma[i] = ma / 255.0
				te_angle[i] = angle
		'''
		'''
		cnt = contours[0]
		te_area[i] = cv2.contourArea(cnt)
		te_per[i] = cv2.arcLength(cnt,True)
		M = cv2.moments(cnt)

		if M['m10'] != 0:
			te_cx[i] =  int(M['m10'] / M['m00'])
			te_cy[i] =  int(M['m01'] / M['m00'])
		else: 
			#print i
			#plt.imshow(im, cmap = 'binary')
			#plt.show()
			#plt.imshow(te_list[i], cmap = 'binary')
			#plt.show()
			print cnt
			count += 1
			stop 

		hull = cv2.convexHull(cnt,returnPoints = False)
		defects = cv2.convexityDefects(cnt,hull)
		all_defects = 0
		if hull.all() != 0:
			for j in range(len(defects)):
				all_defects += defects[j][0][3]	
			all_defects /= len(defects)
			te_hull[i] = all_defects
		else:
			te_hull[i] = all_defects
		'''

	print arr2
	print '\n'
	print count 

	tr = np.zeros((len(temp), train.shape[1]))
	trm = np.zeros((len(temp), tr_mom.shape[1]))
	tryy = np.zeros(len(temp))
	for i in range(len(temp)):
		tr[i] = temp[i]
		trm[i] = temp2[i]
		tryy[i] = temp3[i]

	train_mod = np.concatenate((train, tr_mom), axis = 1)
	test_mod = np.concatenate((test, te_mom), axis = 1) 

	return train_mod, test_mod, y

def apply_PCA(train, test, tr_mod_list, te_mod_list, max_dim, ids, labels, vis_PCA = False, tsne = True):
	'''
	Use PCA to create lower dimensional images that can be used with the pre-extracted features

	Parameters:
	te_mod_list - list of the resized training images
	tr_mod_list - list of the resized testing images
	train - 2D array of pre-extracted features for the training set
	test - 2D array of pre-extracted features for the testing set
	ids - 1D array that holds a label for each species ranging from 0 to n - 1
	labels - 1D array of the name of each leaf species

	Return:
	train - 2D array of pre-extracted features for the training set with the flattened PCA features
	test - 2D array of pre-extracted features for the testing set with the flattened PCA features
	'''
	print tr_mod_list.shape
	print te_mod_list.shape

	# Images need to be flattened to apply PCA to them
	print 'Applying PCA...'
	tr_flat = np.zeros((len(tr_mod_list), max_dim * max_dim))
	for i in range(len(tr_mod_list)):
		tr_flat[i] = tr_mod_list[i].ravel()

	te_flat = np.zeros((len(te_mod_list), max_dim * max_dim))
	for i in range(len(te_mod_list)):
		te_flat[i] = te_mod_list[i].ravel()
	
	pca = PCA(n_components = 30)	

	# Fit PCA to our training images and transform our training and tests images
	pca.fit(tr_flat)
	print 'Number of components for PCA:', pca.n_components_
	tr_flat_pca = pca.transform(tr_flat)
	te_flat_pca = pca.transform(te_flat)
	
	train = np.concatenate((train, tr_flat_pca), axis = 1)
	test = np.concatenate((test, te_flat_pca), axis = 1)	


	# Display a plot after t-SNE is applied to look at the relationships between leaves
	if tsne:
		# If we want to use the coordinates for each leaf from t-SNE
		t_pca_all = np.concatenate((tr_flat_pca, te_flat_pca), axis = 0)
		t_mod_all = np.concatenate((tr_mod_list, te_mod_list))

		# Using t-SNE to spot patterns in the data	
		tsne = TSNE(n_components = 2, perplexity = 40.0, verbose = 1)
		tsne_result = tsne.fit_transform(t_pca_all)
		tsne_scaled = StandardScaler().fit_transform(tsne_result)

		# visualize_tsne works only for the labeled images (training set)
		# visualize.visualize_tsne(tsne_scaled, ids, labels)
		x_pt, y_pt = visualize.visualize_tsne_images(tsne_scaled, t_mod_all)

		train = np.concatenate((train, x_pt[0: len(train)], y_pt[0: len(train)]), axis = 1)
		test = np.concatenate((test, x_pt[len(train): len(test) + len(train)],
				y_pt[len(train): len(test) + len(train)]), axis = 1)
		

	# Display what the PCA leaves look like
	if vis_PCA:
		visualize.visualize_PCA(tr_flat_pca, tr_flat, pca)

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
