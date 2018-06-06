# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import random
import math

####### Global variables #######
global_visible_size = 64
global_hidden_size = 25
global_lambda = 1
global_beta = 1
global_rho = 0.05              # Sparsity parameter

####### Definitions #######
# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Regularized cost function
def reg_cost(theta, arr_x, arr_y):
	# Change our weights and bias values back into their original shape
	arr_W1, arr_W2, arr_b1, arr_b2 = reshape(theta)

	# Find our hypothesis
	h, a2 = feedforward(arr_W1, arr_W2, arr_b1, arr_b2, arr_x)

	# Find the average activation of each hidden unit averaged over the training set
	rho_hat = (1.0 / m) * np.sum(a2, axis = 0)         # (25,) vector

	# Calculate the cost
	KL_divergence = global_beta * np.sum((global_rho * np.log(global_rho / rho_hat) + (1 - global_rho) * np.log((1 - global_rho) / (1 - rho_hat))))

	# Calculate the cost
	cost1 = (1.0 / (2 * m)) * np.sum(np.multiply((h - arr_y), (h - arr_y)))
	cost2 = (global_lambda / (2.0)) * (np.sum(np.multiply(arr_W1, arr_W1)))
	cost3 = (global_lambda / (2.0)) * (np.sum(np.multiply(arr_W2, arr_W2)))
	cost = cost1 + cost2 + cost3 + KL_divergence
		
	return cost

# Feedforward
def feedforward(W1, W2, b1, b2, arr_x):

	'''
	We will be running our sigmoid function twice.
	Tile function allows us to duplicate our rows to the proper dimensions without requiring a for loop.
	This enables each row in our dot product to receive the same bias term. If it were a (25, 10000) array it is equivalent to adding 
	our bias column to each dot product column with just + b1 (since b1 starts as a column).
	'''
	a2 = sigmoid(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))    # (10000, 25) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, W2.T) + np.tile(np.ravel(b2), (m, 1)))       # (10000, 64) matrix

	return a3, a2

# Backpropagation
def backprop(theta, arr_x, arr_y):
	# Change our weights and bias values back into their original shape
	arr_W1, arr_W2, arr_b1, arr_b2 = reshape(theta)

	a3, a2 = feedforward(arr_W1, arr_W2, arr_b1, arr_b2, arr_x)

	# Find the average activation of each hidden unit averaged over the training set
	rho_hat = np.mean(a2, axis = 0)         # (25,) vector
	rho_hat = np.tile(rho_hat, (m, 1))      # (10000, 25) matrix (Could just leave it as a vector, code still runs the same)

	delta3 = np.multiply(a3 - arr_y, a3 * (1 - a3))   # (10000, 64)
	delta2 = np.multiply(np.dot(delta3, arr_W2) + global_beta * (-(global_rho / rho_hat) + ((1 - global_rho) / (1 - rho_hat))), a2 * (1 - a2))
	# delta2 is a (10000, 25) matrix
	
	# Compute the partial derivatives
	pd_W1 = np.dot(delta2.T, arr_x)  # (25, 64)
	pd_W2 = np.dot(delta3.T, a2)     # (64, 25)
	pd_b1 = np.sum(delta2, axis = 0) # (25,) vector
	pd_b2 = np.sum(delta3, axis = 0) # (64,) vector

	'''
	# Computing the batch gradient
	del_W1 = np.zeros((len(pd_W1), len(pd_W1[0])))
	del_W2 = np.zeros((len(pd_W2), len(pd_W2[0])))
	del_b1 = np.zeros((len(pd_b1), 1))
	del_b2 = np.zeros((len(pd_b2), 1))
	'''

	del_W1 = (1.0 / m) * pd_W1
	del_W2 = (1.0 / m) * pd_W2
	del_b1 = (1.0 / m) * pd_b1
	del_b2 = (1.0 / m) * pd_b2
		
	'''
	# FIXME REMOVE THIS SECTION
	arr_W1 = arr_W1 - global_alpha * ((1.0 / m) * del_W1 + global_lambda * arr_W1)
	arr_W2 = arr_W2 - global_alpha * ((1.0 / m) * del_W2 + global_lambda * arr_W2)
	arr_b1 = arr_b1 - global_alpha * ((1.0 / m) * del_b1)
	arr_b2 = arr_b2 - global_alpha * ((1.0 / m) * del_b2)
	'''

	# Changed the gradients into a one dimensional vector
	del_W1 = np.ravel(del_W1)
	del_W2 = np.ravel(del_W2)
	D_vals = np.concatenate((del_W1, del_W2, del_b1, del_b2))

	return D_vals

# Set up our weights and bias terms
def weights_bias():
	# Initialize parameters randomly based on layer sizes.
	# We'll choose weights uniformly from the interval [-r, r]
	r  = 0.12
	# math.sqrt(6) / math.sqrt(global_hidden_size + global_visible_size + 1);
	random_weight1 = np.random.rand(global_hidden_size, global_visible_size)     # (25, 64) matrix
	random_weight1 = random_weight1 * 2 * r - r
	random_weight2 = np.random.rand(global_visible_size, global_hidden_size)     # (64, 25) matrix      
	random_weight2 = random_weight2 * 2 * r - r

	# Set up our bias term
	bias1 = np.random.rand(global_hidden_size, 1)     # (25, 1) matrix
	bias1 = bias1 * 2 * r - r
	bias2 = np.random.rand(global_visible_size, 1)    # (64, 1) matrix
	bias2 = bias2 * 2 * r - r

	# Combine these into a 1-dimension vector
	random_weight1_1D = np.ravel(random_weight1)
	bias1_1D = np.ravel(bias1)
	random_weight2_1D = np.ravel(random_weight2)
	bias2_1D = np.ravel(bias2)

	# Create a vector theta = W1 + W2 + b1 + b2
	theta_vals = np.concatenate((random_weight1_1D, random_weight2_1D, bias1_1D, bias2_1D))	
	
	return theta_vals

# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:global_hidden_size * global_visible_size], (global_hidden_size, global_visible_size))
	W2 = np.reshape(theta[global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size], (global_visible_size, global_hidden_size))
	b1 =np.reshape(theta[2 * global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size + global_hidden_size], (global_hidden_size, 1))
	b2 =np.reshape(theta[2 * global_hidden_size * global_visible_size + global_hidden_size: len(theta)], (global_visible_size, 1))
	
	return W1, W2, b1, b2

####### Generate training set #######

# Extract the provided data. We need to use scipy since the data is in a matlab file format
images_data = scipy.io.loadmat('starter/IMAGES.mat')

# The images that we need to extract are under the name 'IMAGES'
images = images_data['IMAGES']

patch_size = 8       # We want to use 8x8 patches
num_patches = 100    # Total number of patches we will have

# Set up an array of zeros for the patches (64, 10000)
patches = np.zeros((patch_size ** 2, num_patches))

# Let's look at one of the images
pick_image = 0
pick_image = int(input('Enter digit representing an image (0-9):'))
while (pick_image > 9 or pick_image < 0):
	pick_image = int(input('Please pick a digit in the range 0-9:'))


image = images[:,:, pick_image]
'''
plt.imshow(image, cmap = 'binary')
plt.show()
'''

# Now we want to break the image up into patches
for i in range(len(patches[0])):
	int_random = random.randint(0, 504)
	temp = image[int_random: int_random + 8, int_random: int_random + 8]
	temp = np.reshape(temp, (64, 1))
	patches[:, i:i+1] = temp

'''
# Check to make sure our code is running correctly
image2 = np.reshape(patches[:,0:1], (8, 8))
plt.imshow(image2, cmap = 'binary', interpolation = 'none')
plt.show()
'''

####### Sparse autoencoder objective #######
m = len(patches[0])

# We need to values in patches to range from 0 to 1
old_min = -1
old_max = 1
new_min = 0
new_max = 1
patches = ((patches - old_min) / (old_max - old_min)) * (old_max - new_min) + new_min

# Tranpose patches to the dimension we want
patches = patches.T                        # (10000, 64)
y = patches

# Create our weights and bias terms
theta1 = weights_bias()

'''
# Check that our cost function is working
cost_test = reg_cost(theta1, patches, y)
print cost_test
# We had a cost value of 21.7
'''

# Gradient checking from scipy to see if our backprop function is working properly. Theta_vals needs to be a 1-D vector.
print scipy.optimize.check_grad(reg_cost, backprop, theta1, patches, y)
# Recieved a value of N/A
