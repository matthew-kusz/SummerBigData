# Import the necessary packages
import numpy as np
import scipy.optimize
import scipy.io
import cnn_convolve

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

# Now we need to load in the STL10 images
data = scipy.io.loadmat('provided_data/stlTrainSubset.mat')
train_labels = data['trainLabels']
train_images = data['trainImages']
num_train_images = int(data['numTrainImages'])
print train_labels.shape
print train_images.shape
print num_train_images

# Grab a small amount of images to test our convolve code on
train_images = train_images[:,:,:,0:8]
print train_images.shape

convolved_features = cnn_convolve.convolve(global_image_dim, global_hidden_size, train_images, W1_final, b1_final, ZCA_matrix, mean_patches)
