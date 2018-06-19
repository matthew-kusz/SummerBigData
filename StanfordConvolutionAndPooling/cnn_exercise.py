# Import the necessary packages
import numpy as np
import scipy.optimize
import scipy.io
import cnn_convolve, cnn_pooling
import random
import sys

####### Global variables #######
global_step = 0
global_image_dim = 64
global_image_channels = 3
global_patch_dim = 8
global_num_patches = 50000
global_visible_size = global_patch_dim * global_patch_dim * global_image_channels
global_output_size = global_visible_size
global_hidden_size = 400
global_epsilon = 0.1
global_pool_dim = 19

####### Definitions #######
# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Feedforward
def feedforward(W1, W2, b1, b2, arr_x, m):

	'''
	We will be running our sigmoid function twice.
	Tile function allows us to duplicate our rows to the proper dimensions without requiring a for loop.
	This enables each row in our dot product to receive the same bias term. If it were a (25, 10000) array it is equivalent to adding 
	our bias column to each dot product column with just + b1 (since b1 starts as a column).
	'''
	a2 = sigmoid(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))    # (m, 400) matrix

	# Second run
	a3 = np.dot(a2, W2.T) + np.tile(np.ravel(b2), (m, 1))                # (m, 192) matrix

	return a3, a2

# Checking the convolution code
def check_conv(im, conv_feat, mu, ZCA, W1, b1):
	m = len(im)
	for i in range(1000):
		# Pick a random data patch
		feature_num = np.random.randint(0, global_hidden_size - 1)
		image_num = np.random.randint(0, 7)
		image_row = np.random.randint(0, global_image_dim - global_patch_dim)
		image_col = np.random.randint(0, global_image_dim - global_patch_dim)
	
		patch = im[image_num, image_row: image_row + global_patch_dim, image_col: image_col + global_patch_dim, :]
		# Flatten the image in groups of color
		patch = np.concatenate((patch[:,:,0].flatten(), patch[:,:,1].flatten(), patch[:,:,2].flatten()))
		patch -= np.ravel(mean_patches)
		patch = patch.reshape(1,192)
		patch = np.dot(patch, ZCA_matrix)
		
		features = sigmoid(np.dot(patch, W1.T) + np.tile(np.ravel(b1), (m, 1)))
		
		if (abs(features[0, feature_num] - conv_feat[feature_num, image_num, image_row, image_col]) > 1e-9):
			print 'Convolved feature does not match activation from autoencoder'
			print 'Feature Number    : ', feature_num
			print 'Image Number      : ', image_num
			print 'Image Row         : ', image_row
			print 'Image Column      : ', image_col
			print 'Convolved feature : ', conv_feat[feature_num, image_num, image_row, image_col]
			print 'Sparse AE feature : ', features[1, feature_num]
			print 'Error: Convolved feature does not match activation from autoencoder'
			sys.exit()
		
	print 'Congratz! Your convolution code passed the test.'

	return	

# Generate training data
def gen_train_data():
	data = scipy.io.loadmat('provided_data/stlTrainSubset.mat')
	labels = data['trainLabels']
	images = data['trainImages']
	num_images = int(data['numTrainImages'])

	# reformat our images array
	ord_img = np.zeros((2000, 64, 64, 3))
	for i in range(2000):
		ord_img[i] = images[:, :, :, i]

	return labels, ord_img, num_images

# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:global_hidden_size * global_visible_size], (global_hidden_size, global_visible_size))
	W2 = np.reshape(theta[global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size], (global_visible_size, global_hidden_size))
	b1 =np.reshape(theta[2 * global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size + global_hidden_size], (global_hidden_size, 1))
	b2 =np.reshape(theta[2 * global_hidden_size * global_visible_size + global_hidden_size: len(theta)], (global_visible_size, 1))
	
	return W1, W2, b1, b2

####### Code #######
# First we need to grab the theta values, ZCA_matrix and mean patches we obtained from our linear decoder
final_theta = np.genfromtxt('provided_data/finalWeightsRho0.035Lambda0.003Beta5.0Size100000HL400.out')
ZCA_matrix = np.genfromtxt('provided_data/ZCAwhitening0.035Lambda0.003Beta5.0Size100000HL400.out')
mean_patches = np.genfromtxt('provided_data/meanPatchesRho0.035Lambda0.003Beta5.0Size100000HL400.out')

# We need to reshape our final_theta values and only use W1 and b1
W1_final, W2_final, b1_final, b2_final = reshape(final_theta)

# We need to also reshape our ZCA_matrix and mean_patches
ZCA_matrix = np.reshape(ZCA_matrix, (192, 192))
mean_patches = np.reshape(mean_patches, (1, 192))

# Generate our training data
train_labels, train_images, num_train_images = gen_train_data()

# Grab a small amount of images to test our convolve code on
train_test_images = train_images[0:8, :, :, :]

# Implement convolution
convolved_features = cnn_convolve.convolve(global_patch_dim, global_hidden_size, train_test_images, W1_final, b1_final, ZCA_matrix, mean_patches)

# We will now check our convolution
check_conv(train_test_images, convolved_features, mean_patches, ZCA_matrix, W1_final, b1_final)

# Implement pooling
pooled_features = cnn_pooling.pooling(global_pool_dim, convolved_features)
