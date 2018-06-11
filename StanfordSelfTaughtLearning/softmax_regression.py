# Import the necessary pachages
import struct as st
import gzip
import scipy.io
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import grab_data.py

####### Global variables #######
global_step = 0
global_input_size  = 28 * 28
global_hidden_size = 196
global_lambda = 1e-4;       # weight decay parameter        

# Sigmoid function
def sigmoid(value):
	return (np.exp(value) / np.sum(np.exp(value))

# Regularized cost function
def reg_cost(theta, arr_x, arr_y):	
	# Reshape our thetas back into their original shape  
	arr_theta1 = np.reshape(theta[0:25 * n], (25, n))
	arr_theta2 = np.reshape(theta[25 * n: len(theta)], (10, 26))

	# Find our hypothesis
	h, nano = feedforward(arr_theta1, arr_theta2, arr_x)

	# Calculate the cost
	first_half = np.multiply(arr_y, np.log(h))
	second_half = np.multiply((1.0 - arr_y), np.log(1.0 - h))

	cost1 = np.sum((-1.0 / m) * (first_half + second_half))
	cost2 = (lambda1 / (2.0 * m)) * (np.sum(arr_theta1 ** 2) - np.sum(arr_theta1[:,0] ** 2))
	cost3 = (lambda1 / (2.0 * m)) * (np.sum(arr_theta2 ** 2) - np.sum(arr_theta2[:,0] ** 2))
	cost = cost1 + cost2 + cost3

	# Incase we get booted early
	if (global_iterations % 20 == 0):
		print cost
		np.savetxt(file_name, theta, delimiter = ',')

	return cost

# Feedforward
def feedforward(theta1, theta2, arr_x):

	# We will be running our sigmoid function twice
	a2 = sigmoid(np.dot(arr_x, theta1.T))	

	# Add a column of ones to our array of a2
	arr_ones = np.ones((m, 1))
	a2 = np.hstack((arr_ones, a2))        # (m, 197) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, theta2.T))    # (m, 5) matrix

	return a3, a2

# Backpropagation
def backprop(theta, arr_x, arr_y_train):
	# Change our theta values back into their original shape
	arr_theta1 = np.reshape(theta[0:25 * n], (25, n))
	arr_theta2 = np.reshape(theta[25 * n: len(theta)], (10, 26))

	a3, a2 = feedforward(arr_theta1, arr_theta2, arr_x)
	Delta2 = 0
	Delta1 = 0
	'''
	# To keep track of our iterations
	global global_step
	global_step += 1
	if (global_step % 50 == 0):
		np.savetxt(filename, theta, delimiter = ',')
		print 'Global step: %g' %(global_step)

	# Change our weights and bias values back into their original shape
	arr_W1, arr_W2 = reshape(theta)
	
	a3, a2 = feedforward(arr_W1, arr_W2, arr_x)

	delta3 = np.multiply(-(arr_y - a3), a3 * (1 - a3))   # (m, 784)
	delta2 = np.multiply(np.dot(delta3, arr_W2)          # (m, 196)
	
	# Compute the partial derivatives
	pd_W1 = np.dot(delta2.T, arr_x)  # (196, 784)
	pd_W2 = np.dot(delta3.T, a2)     # (784, 197)
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
	'''
	for i in range(len(arr_x)):
		delta3 = a3[i] - arr_y_train[i]                   # Vector length 10
		
		delta3 = np.reshape(delta3, (len(delta3), 1))     # (10, 1) matrix
		temp_a2 = np.reshape(a2[i], (len(a2[i]), 1))      # (26, 1) matrix
		delta2 = np.multiply(np.dot(arr_theta2.T, delta3), temp_a2 * (1 - temp_a2))

		Delta2 = Delta2 + np.dot(delta3, temp_a2.T)       # (10, 26) matrix
		temp_arr_x = np.reshape(arr_x[i], (len(arr_x[i]), 1))
		
		# We need to remove delta2[0] from Delta1
		Delta1 = Delta1 + np.delete(np.dot(delta2, temp_arr_x.T), 0, axis = 0) # (25, 401) matrix
		
	# Compute the unregularized gradient
	D1_temp = (1.0 / m) * Delta1 
	D2_temp = (1.0 / m) * Delta2
	
	# Compute the regularized gradient
	D1 = (1.0 / m) * Delta1 + lambda1 * arr_theta1
	D1[0:, 0:1] = D1_temp[0:, 0:1]    # (196, 784) matrix
	D2 = (1.0 / m) * Delta2 + lambda1 * arr_theta2
	D2[0:, 0:1] = D2_temp[0:, 0:1]    # (5, 197) matrix

	# Changed the gradient into a one dimensional vector
	D1 = np.ravel(D1)
	D2 = np.ravel(D2)
	D_vals = np.concatenate((D1, D2))

	return D_vals

# Set up our weights and bias terms
def weights_bias():
	# Initialize parameters randomly based on layer sizes.
	# We'll choose weights uniformly from the interval [-r, r]
	r  = 0.12
	# math.sqrt(6) / math.sqrt(global_hidden_size + global_visible_size + 1);
	random_weight1 = np.random.rand(global_hidden_size, global_input_size)   # (196, 784) matrix
	random_weight1 = random_weight1 * 2 * r - r
	random_weight2 = np.random.rand(5, global_hidden_size + 1)               # (5, 197) matrix      
	random_weight2 = random_weight2 * 2 * r - r

	# Combine these into a 1-dimension vector
	random_weight1_1D = np.ravel(random_weight1)
	random_weight2_1D = np.ravel(random_weight2)

	# Create a vector theta = W1 + W2
	theta_vals = np.concatenate((random_weight1_1D, random_weight2_1D))	
	
	return theta_vals

# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:global_hidden_size * global_input_size], (global_hidden_size, global_input_size))
	W2 = np.reshape(theta[global_hidden_size * global_input_size: 2 * global_hidden_size * 5], (5, global_hidden_size))

	return W1, W2

# Change the file name here
file_name = 'outputs/'

# Set up how large we want our data set (max of 60,000)
size = 100

# Import the data we want
time_start = time.time()
train , labels_train = grab_data.get_data(size, '04')
time_finish = time.time()

# Add a column of ones to our array of x_vals
m = len(x_vals)                               # Number of training examples (rows)
n = len(x_vals[0])   			      # Number of columns
arr_ones = np.ones((m, 1))
x_vals = np.hstack((arr_ones, x_vals))        # (m, 784) matrix

# Set up an array that will be either 1 or 0 depending on which number we are looking at
y_vals_train = np.zeros((len(y_vals), 5))

for i in range(10):
	# Set up an array with the values that stand for each number
	arr_num = [0, 1, 2, 3, 4]
	
	for j in range(len(y_vals)):
		if (y_vals[j] == arr_num[i]):
			y_vals_train[j, i:] = 1
		
		else:
			y_vals_train[j, i:] = 0


# Randomly initialize our theta values in a range [-0.12, 0.12]
theta1 = weights_bias()

# Gradient checking from scipy to see if our backprop function is working properly. Theta_vals needs to be a 1-D vector.
print scipy.optimize.check_grad(reg_cost, backprop, theta1, x_vals, y_vals_train)
# Recieved a value of 4.95e-06

'''
initial_cost = reg_cost(theta_vals, x_vals, y_vals_train, lambda1)
print "Our initial cost value is %g." %(initial_cost)
	
# Use scipys minimize function to compute the theta values
minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta1, method = 'L-BFGS-B', tol = 1e-4, jac = backprop, args = (x_vals, y_vals_train))#, options = {'disp': True}

# Set up the new theta values we hav from our minimize function
theta_new = minimum.x

final_cost = reg_cost(theta_new, x_vals, y_vals_train, lambda1)
print "Our final cost value is %g." %(final_cost)

# Save the theta values to use later
np.savetxt(file_name, theta_new, delimiter = ',')
'''
