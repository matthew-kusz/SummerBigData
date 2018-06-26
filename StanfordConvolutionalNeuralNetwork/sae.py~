# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import random
import math
import argparse

####### Global variables #######
parser = argparse.ArgumentParser()
parser.add_argument('Lambda', help = 'Adjust to prevent overfitting.', type = float)
parser.add_argument('Rho', help = 'Desired average activation of the hidden units (sparsity parameter)', type = float)
parser.add_argument('Beta', help = 'Weight of sparsity penalty term', type = float)

args = parser.parse_args()

global_patch_dim = 15
global_visible_size = global_patch_dim ** 2
global_hidden_size = 100
global_rho = args.Rho;                # desired average activation of the hidden units (sparsity parameter). (0.01)
global_lambda = args.Lambda;          # weight decay parameter (1e-3)
global_beta = args.Beta;              # weight of sparsity penalty term (1)

print 'You chose', args

filename = 'outputs/finalWeightsMNISTSize10000Patches15x15L' + str(global_lambda) + 'B' + str(global_beta) + 'Rho' + str(global_rho) + 'HL' + str(global_hidden_size) + '.out'

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
	rho_hat = (1.0 / m) * np.sum(a2, axis = 0)         # (100,) vector

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
	a2 = sigmoid(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))    # (10000, 100) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, W2.T) + np.tile(np.ravel(b2), (m, 1)))       # (10000, 225) matrix

	return a3, a2

# Backpropagation
global_step = 0
def backprop(theta, arr_x, arr_y):
	# Change our weights and bias values back into their original shape
	arr_W1, arr_W2, arr_b1, arr_b2 = reshape(theta)

	a3, a2 = feedforward(arr_W1, arr_W2, arr_b1, arr_b2, arr_x)

	# Find the average activation of each hidden unit averaged over the training set
	rho_hat = np.mean(a2, axis = 0)         # (100,) vector
	rho_hat = np.tile(rho_hat, (m, 1))      # (10000, 100) matrix (Could just leave it as a vector, code still runs the same)

	delta3 = np.multiply(a3 - arr_y, a3 * (1 - a3))   # (10000, 225)
	delta2 = np.multiply(np.dot(delta3, arr_W2) + global_beta * (-(global_rho / rho_hat) + ((1 - global_rho) / (1 - rho_hat))), a2 * (1 - a2))
	# delta2 is a (10000, 25) matrix
	
	# Compute the partial derivatives
	pd_W1 = np.dot(delta2.T, arr_x)   # (100, 225)
	pd_W2 = np.dot(delta3.T, a2)      # (225, 100)
	pd_b1 = np.mean(delta2, axis = 0) # (25,) vector
	pd_b2 = np.mean(delta3, axis = 0) # (225,) vector

	del_W1 = (1.0 / m) * pd_W1 + global_lambda * arr_W1
	del_W2 = (1.0 / m) * pd_W2 + global_lambda * arr_W2
	del_b1 = pd_b1
	del_b2 = pd_b2

	# Change the gradients into a one dimensional vector
	del_W1 = np.ravel(del_W1)
	del_W2 = np.ravel(del_W2)
	D_vals = np.concatenate((del_W1, del_W2, del_b1, del_b2))

	return D_vals

# Set up our weights and bias terms
def weights_bias():
	# Initialize parameters randomly based on layer sizes.
	# We'll choose weights uniformly from the interval [-r, r]
	r  = 0.12

	# Generate a seed so our random values remain the same through each run
	np.random.seed(7)

	random_weight1 = np.random.rand(global_hidden_size, global_visible_size)     # (100, 225) matrix
	random_weight1 = random_weight1 * 2 * r - r
	random_weight2 = np.random.rand(global_visible_size, global_hidden_size)     # (225, 100) matrix      
	random_weight2 = random_weight2 * 2 * r - r

	# Set up our bias term
	bias1 = np.random.rand(global_hidden_size, 1)     # (100, 1) matrix
	bias1 = bias1 * 2 * r - r
	bias2 = np.random.rand(global_visible_size, 1)    # (225, 1) matrix
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
	W2 = np.reshape(theta[global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size],
		(global_visible_size, global_hidden_size))
	b1 =np.reshape(theta[2 * global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size + global_hidden_size],
		(global_hidden_size, 1))
	b2 =np.reshape(theta[2 * global_hidden_size * global_visible_size + global_hidden_size: len(theta)], (global_visible_size, 1))
	
	return W1, W2, b1, b2

####### Code #######
# Grab the patches that we need
patches = np.genfromtxt('outputs/MNIST10000patches15x15.out')  # (10000, 225)
patches = np.reshape(patches, (len(patches) / global_visible_size, global_visible_size))
print patches.shape
m = len(patches)

# We need the values in patches to range from 0 to 1
old_min = -1
old_max = 1
new_min = 0
new_max = 1
patches = ((patches - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

y = patches

# Create our weights and bias terms
theta1 = weights_bias()

'''
# FOR CHECKING OUR COST FUNCTION AND BACKPROP
# Check that our cost function is working
patches = patches[0:10]
print patches.shape
y = patches
m = len(patches)
cost_test = reg_cost(theta1, patches, y)
print cost_test
# We had a cost value of 44.7
# Gradient checking from scipy to see if our backprop function is working properly. Theta_vals needs to be a 1-D vector.
print scipy.optimize.check_grad(reg_cost, backprop, theta1, patches, y)
# Recieved a value of 2.8e-5
'''

print 'Cost before minimization: %g' %(reg_cost(theta1, patches, y))

# Minimize the cost value
minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta1, method = 'CG', tol = 1e-4, jac = backprop, args = (patches, y), options = {"disp":True})
theta_new = minimum.x

print 'Cost after minimization: %g' %(reg_cost(theta_new, patches, y))

# Save to a file to use later
np.savetxt(filename, theta_new, delimiter = ',')
