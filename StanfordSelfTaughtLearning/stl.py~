# Import the necessary packages
import struct as st
import gzip
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import random
import math
import grab_data
import time

####### Global variables #######
global_step = 0
global_input_size  = 28 * 28
global_num_labels  = 5
global_hidden_size = 196
global_rho = 0.1;           # desired average activation of the hidden units (sparsity parameter).
global_lambda = 3e-3;       # weight decay parameter       
global_beta = 3;            # weight of sparsity penalty term   

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
	a2 = sigmoid(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))    # (m, 200) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, W2.T) + np.tile(np.ravel(b2), (m, 1)))       # (m, 784) matrix

	return a3, a2

# Backpropagation
def backprop(theta, arr_x, arr_y):
	# To keep track of our iterations
	global global_step
	global_step += 1
	if (global_step % 50 == 0):
		np.savetxt(filename, theta, delimiter = ',')
		print 'Global step: %g' %(global_step)

	# Change our weights and bias values back into their original shape
	arr_W1, arr_W2, arr_b1, arr_b2 = reshape(theta)
	
	a3, a2 = feedforward(arr_W1, arr_W2, arr_b1, arr_b2, arr_x)

	# Find the average activation of each hidden unit averaged over the training set
	rho_hat = np.mean(a2, axis = 0)         # (200,) vector
	rho_hat = np.tile(rho_hat, (m, 1))      # (m, 200) matrix (Could just leave it as a vector, code still runs the same)

	delta3 = np.multiply(-(arr_y - a3), a3 * (1 - a3))   # (m, 784)
	delta2 = np.multiply(np.dot(delta3, arr_W2) + global_beta * (-(global_rho / rho_hat) + ((1 - global_rho) / (1 - rho_hat))), a2 * (1 - a2))
	# delta2 is a (m, 25) matrix
	
	# Compute the partial derivatives
	pd_W1 = np.dot(delta2.T, arr_x)  # (200, 784)
	pd_W2 = np.dot(delta3.T, a2)     # (784, 25)
	pd_b1 = np.mean(delta2, axis = 0) # (200,) vector
	pd_b2 = np.mean(delta3, axis = 0) # (784,) vector

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
	random_weight1 = np.random.rand(global_hidden_size, global_input_size)     # (200, 784) matrix
	random_weight1 = random_weight1 * 2 * r - r
	random_weight2 = np.random.rand(global_input_size, global_hidden_size)     # (784, 200) matrix      
	random_weight2 = random_weight2 * 2 * r - r

	# Set up our bias term
	bias1 = np.random.rand(global_hidden_size, 1)     # (200, 1) matrix
	bias1 = bias1 * 2 * r - r
	bias2 = np.random.rand(global_input_size, 1)    # (784, 1) matrix
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
	W1 = np.reshape(theta[0:global_hidden_size * global_input_size], (global_hidden_size, global_input_size))
	W2 = np.reshape(theta[global_hidden_size * global_input_size: 2 * global_hidden_size * global_input_size], (global_input_size, global_hidden_size))
	b1 =np.reshape(theta[2 * global_hidden_size * global_input_size: 2 * global_hidden_size * global_input_size + global_hidden_size], (global_hidden_size, 1))
	b2 =np.reshape(theta[2 * global_hidden_size * global_input_size + global_hidden_size: len(theta)], (global_input_size, 1))
	
	return W1, W2, b1, b2

time_start = time.time()
# Import the file we want
train , labels_train = grab_data.get_data(60000, '59')
time_finish = time.time()
print 'Total time for obtaining data = %g' %(time_finish - time_start)

# Set up the filename we want to use
filename = 'outputs/finalWeightsL3e-3B3Rho0.1Size60000HL196.out'

# Need to know how many inputs we have
m = len(train)

# Create our weights and bias terms
theta1 = weights_bias()

# We want out x = y for our sparse autoencoder
y = train

'''
# Check that our cost function is working
cost_test = reg_cost(theta1, train, y)
print cost_test
# We had a cost value of 38 (from 20 nodes instead of 200)

# Gradient checking from scipy to see if our backprop function is working properly. Theta_vals needs to be a 1-D vector.
print scipy.optimize.check_grad(reg_cost, backprop, theta1, train, y)
# Recieved a value of 6e-5
'''

print 'Cost before minimization: %g' %(reg_cost(theta1, train, y))
time_start2 = time.time()

# Minimize the cost value
minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta1, method = 'L-BFGS-B', tol = 1e-4, jac = backprop, args = (train, y)) #options = {"disp":True}
print minimum
theta_new = minimum.x

print 'Cost after minimization: %g' %(reg_cost(theta_new, train, y))
time_finish2 = time.time()

# Save to a file to use later
np.savetxt(filename, theta_new, delimiter = ',')

print 'Total time for minimization = %g' %(time_finish2 - time_start2)

