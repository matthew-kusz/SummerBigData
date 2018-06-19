# Import the necessary packages
import numpy as np

####### Definitions #######
def convolve(patch_dim, num_features, images, W, b, ZCA_white, mean_patch):
	# Number of images
	m = images.shape[3]
	# Dimension of image
	img_dim = images.shape[0]
	# Number of channels
	num_channel = images.shape[2]
	print m, img_dim, num_channel
	
	convolved_features = np.zeros((num_features, m, img_dim - patch_dim + 1, img_dim - patch_dim + 1))

	print images.shape
	'''
	# Precompute the matrices that will be used during the convolution
	images -= 
	for i in m:
		for j in num_features:
		
		# convolution of image with feature matrix for each channel
		convolved_image = np.zeros(img_dim - patch_dim + 1, img_dim - patch_dim + 1)
		for k in num_channel:
			# Obtain the feature (patch_dim x patch_dim) needed during the convolution
			feature = np.zeros((patch_dim, patch_dim))

			# Flip the feature matrix because of the definition of convolution, as explained later
			feature = np.flipud(?)
	'''
