# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import random
import math

####### Global variables #######
filename = 'outputs/finalWeightsL3e-3B5Rho0.035TESTING.out'
global_step = 0
global_image_channels = 3
global_patch_dim = 8
global_visible_size = global_patch_dim * global_patch_dim * global_image_channels
global_hidden_size = 400
global_output_size = global_visible_size
global_lambda = 3e-3
global_beta = 5
global_rho = 0.035             # Sparsity parameter
global_epsilon = 0.1           # ZCA whitening
global_num_patchs = 100000

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
	rho_hat = (1.0 / m) * np.sum(a2, axis = 0)         # (400,) vector

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
	a2 = sigmoid(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))    # (m, 400) matrix

	# Second run
	a3 = np.dot(a2, W2.T) + np.tile(np.ravel(b2), (m, 1))               # (m, 192) matrix

	return a3, a2

# Backpropagation
def backprop(theta, arr_x, arr_y):
	# To keep track of our iterations
	global global_step
	global_step += 1
	if (global_step % 20 == 0):
		np.savetxt(filename, theta, delimiter = ',')
		print 'Global step: %g' %(global_step)
	
	# Change our weights and bias values back into their original shape
	arr_W1, arr_W2, arr_b1, arr_b2 = reshape(theta)

	a3, a2 = feedforward(arr_W1, arr_W2, arr_b1, arr_b2, arr_x)

	# Find the average activation of each hidden unit averaged over the training set
	rho_hat = np.mean(a2, axis = 0)         # (400,) vector
	rho_hat = np.tile(rho_hat, (m, 1))      # (m, 400) matrix (Could just leave it as a vector, code still runs the same)

	delta3 = a3 - arr_y  # (m, 192)
	delta2 = np.multiply(np.dot(delta3, arr_W2) + global_beta * (-(global_rho / rho_hat) + ((1 - global_rho) / (1 - rho_hat))), a2 * (1 - a2))
	# delta2 is a (m, 400) matrix
	
	# Compute the partial derivatives
	pd_W1 = np.dot(delta2.T, arr_x)  # (400, 192)
	pd_W2 = np.dot(delta3.T, a2)     # (192, 400)
	pd_b1 = np.mean(delta2, axis = 0) # (400,) vector
	pd_b2 = np.mean(delta3, axis = 0) # (192,) vector

	del_W1 = (1.0 / m) * pd_W1 + global_lambda * arr_W1
	del_W2 = (1.0 / m) * pd_W2 + global_lambda * arr_W2
	del_b1 = pd_b1
	del_b2 = pd_b2

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
	random_weight1 = np.random.rand(global_hidden_size, global_visible_size)     # (400, 192) matrix
	random_weight1 = random_weight1 * 2 * r - r
	random_weight2 = np.random.rand(global_visible_size, global_hidden_size)     # (192, 400) matrix      
	random_weight2 = random_weight2 * 2 * r - r

	# Set up our bias term
	bias1 = np.random.rand(global_hidden_size, 1)     # (400, 1) matrix
	bias1 = bias1 * 2 * r - r
	bias2 = np.random.rand(global_visible_size, 1)    # (192, 1) matrix
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

data = scipy.io.loadmat('data/stlSampledPatches.mat')
patches = data['patches']
print patches.shape

'''
# We need to values in patches to range from 0 to 1
old_min = -1
old_max = 1
new_min = 0
new_max = 1
patches = ((patches - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
'''

# Tranpose patches to the dimension we want
patches = patches.T                        # (m, 192)
m = len(patches)
# y = patches

# Create our weights and bias terms
theta1 = weights_bias()

'''
FOR CHECKING IF OUR BACKPROP AND COST FUNCTION WORKS
# Check that our cost function is working
cost_test = reg_cost(theta1, patches, y)
print cost_test
# We had a cost value of 81.4

# Gradient checking from scipy to see if our backprop function is working properly. Theta_vals needs to be a 1-D vector.
check_patches = patches[0:100]
print check_patches.shape
y = check_patches
m = len(check_patches)

# Let's apply ZCA whitening
check_patches = check_patches.T
sigma = np.dot(check_patches, check_patches.T) / check_patches[0].shape
u, s, v = np.linalg.svd(sigma)
ZCAWhite = np.multiply(u, np.multiply(np.diag(1.0 / np.sqrt(np.diag(s) + global_epsilon)), u.T))
check_patches = np.dot(ZCAWhite, check_patches)
check_patches = check_patches.T

print scipy.optimize.check_grad(reg_cost, backprop, theta1, check_patches, y)
# Recieved a value of 3.6e-5
'''

check_patches = patches[0:20000]
m = len(check_patches)
# y = check_patches

# Let's apply ZCA whitening
mean_patches = np.mean(check_patches, axis = 0)
mean_patches = np.reshape(mean_patches, (1, check_patches.shape[1]))
check_patches = check_patches - np.tile(mean_patches, (check_patches.shape[0], 1))

sigma = np.dot(check_patches.T, check_patches) / check_patches[0].shape
u, s, v = np.linalg.svd(sigma)
ZCAWhite = np.dot(u, np.dot(np.diag(1.0 / np.sqrt(s + global_epsilon)), u.T))
whitened_patches = np.dot(check_patches, ZCAWhite)
y = whitened_patches

print 'Cost before minimization: %g' %(reg_cost(theta1, whitened_patches, y))

# Minimize the cost value
minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta1, method = 'L-BFGS-B', tol = 1e-4, jac = backprop, args = (whitened_patches, y)) #options = {"disp":True}
theta_new = minimum.x

print 'Cost after minimization: %g' %(reg_cost(theta_new, whitened_patches, y))

# Save to a file to use later
np.savetxt(filename, theta_new, delimiter = ',')

