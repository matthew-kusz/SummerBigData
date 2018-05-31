# Import the necessary pachages
import struct as st
import gzip
import scipy.io
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

# Reading in MNIST data files
def read_images(filename, n=None):
	f = gzip.open(filename)
	f.seek(0)
	magic_num = st.unpack('>4B', f.read(4))
	num_img = st.unpack('>I', f.read(4))[0]
	if not n is None:
		num_img = n
	num_row = st.unpack('>I', f.read(4))[0]
	num_col = st.unpack('>I', f.read(4))[0]
	
	X = np.zeros((num_img, num_row * num_col))	
	total_bytes = num_img * num_row * num_col
	X = np.asarray(
		st.unpack('>' + 'B'*total_bytes, f.read(total_bytes))).reshape((num_img, num_row * num_col))
	f.close()
	return X/255

def read_labels(filename, n=None):
	f = gzip.open(filename)
	f.seek(0)
	magic_num = st.unpack('>4B', f.read(4))
	num_label = st.unpack('>I', f.read(4))[0]
	if not n is None:
		num_label = n
	
	y = np.asarray(
		st.unpack('>' + 'B'*num_label, f.read(num_label)))
	f.close()
	return y	

def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr

# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Regularized cost function
def reg_cost(theta, arr_x, arr_y, lambda1):	
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
	return cost

# Feedforward
def feedforward(theta1, theta2, arr_x):

	# We will be running our sigmoid function twice
	a2 = sigmoid(np.dot(arr_x, theta1.T))	

	# Add a column of ones to our array of a2
	arr_ones = np.ones((m, 1))
	a2 = np.hstack((arr_ones, a2))        # (5000, 26) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, theta2.T))    # (5000, 10) matrix

	return a3, a2

# Backpropagation
def backprop(theta, arr_x, arr_y_train, lambda1):
	# Change our theta values back into their original shape
	arr_theta1 = np.reshape(theta[0:25 * n], (25, n))
	arr_theta2 = np.reshape(theta[25 * n: len(theta)], (10, 26))

	a3, a2 = feedforward(arr_theta1, arr_theta2, arr_x)
	Delta2 = 0
	Delta1 = 0
	
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
	D1 = (1.0 / m) * Delta1 + (lambda1 / float(m)) * arr_theta1
	D1[0:, 0:1] = D1_temp[0:, 0:1]    # (25, 401) matrix
	D2 = (1.0 / m) * Delta2 + (lambda1 / float(m)) * arr_theta2
	D2[0:, 0:1] = D2_temp[0:, 0:1]    # (10, 26) matrix

	# Changed the gradient into a one dimensional vector
	D1 = np.ravel(D1)
	D2 = np.ravel(D2)
	D_vals = np.concatenate((D1, D2))

	return D_vals

def rand_sample(arr_x, arr_y):
	# Combine the x and y arrays so we don't mix up the numbers between them
	arr_xy = np.concatenate((arr_x, arr_y), axis = 1)

	'''
	# Set up our 500 values in order from 0 to 9
	arr_order = np.concatenate((arr_xy[0:50], arr_xy[500:550], arr_xy[1000:1050], arr_xy[1500:1550], arr_xy[2000:2050], arr_xy[2500:2550], arr_xy[3000:3050], arr_xy[3500:3550], arr_xy[4000:4050], arr_xy[4500:4550]), axis = 0)
	'''

	# Randomize these values 
	np.random.shuffle(arr_xy)

	return arr_xy

'''
The x array is under the name 'X', the y array is under the name 'y'
x_vals ia a (5000, 400) array and y_vals is a (5000, 1) array
'''
x_vals = read_images('data/train-images-idx3-ubyte.gz')
y_vals = read_labels('data/train-labels-idx1-ubyte.gz')
y_vals = np.reshape(y_vals, (len(y_vals), 1))
x_vals_modified = x_vals[0:500]
y_vals_modified = y_vals[0:500]

#Set what lambda value we want to use
lambda1 = 1

# Add a column of ones to our array of x_vals
m = len(x_vals_modified)                               # Number of training examples (rows)
arr_ones = np.ones((m, 1))
x_vals_modified = np.hstack((arr_ones, x_vals_modified))        # (5000, 401) matrix

# Set up an array that will be either 1 or 0 depending on which number we are looking at
y_vals_train = np.zeros((len(y_vals), 10))

for i in range(10):
	# Set up an array with the values that stand for each number (10 stands for 0)
	arr_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	
	for j in range(len(y_vals)):
		if (y_vals[j] == arr_num[i]):
			y_vals_train[j, i:] = 1
		
		else:
			y_vals_train[j, i:] = 0

'''
#For testing cost and feedforward
# Extract the provided data. We need to use scipy since the data is in a matlab file format
data_thetas = scipy.io.loadmat('ex4weights.mat')

The array for theta1 is under the name 'Theta1' and the array for theta2 is under the name 'Theta2'
theta1_vals is a (25, 401) array and theta2_vals is a (10, 26) array

theta1_vals = data_thetas['Theta1'] # FIXME
theta2_vals = data_thetas['Theta2'] # FIXME

hypothesis, nan = feedforward(theta1_vals, theta2_vals, x_vals)
J_val = reg_cost(hypothesis, x_vals, y_vals_train, theta1_vals, theta2_vals, lambda1)
print J_val
#J_val = 10.5
#This differs from the value in the intruction sheet
'''

# Randomly initialize our theta values in a range [-0.12, 0.12]
n = len(x_vals[0])   # Number of columns
random_theta1 = np.random.rand(25, n)                        # (25, 401) matrix
random_theta1 = random_theta1 * 2 * 0.12 - 0.12

random_theta2 = np.random.rand(10, len(random_theta1) + 1)   # (10, 26) matrix
random_theta2 = random_theta2 * 2 * 0.12 - 0.12

# Combine these into a 1-dimension vector
random_theta1_1D = np.ravel(random_theta1)   # 10025 1-D vector
random_theta2_1D = np.ravel(random_theta2)   # 260 1_D vector
theta_vals = np.concatenate((random_theta1_1D, random_theta2_1D), axis = 1)


# Gradient checking from scipy to see if our backprop function is working properly. Theta_vals needs to be a 1-D vector.
# print scipy.optimize.check_grad(reg_cost, backprop, theta_vals, x_vals, y_vals_train, lambda1)
# Recieved a value of 4.95e-06


initial_cost = reg_cost(theta_vals, x_vals_modified, y_vals_train, lambda1)
print "Our initial cost value is %g." %(initial_cost)
	
# Use scipys minimize function to compute the theta values
minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta_vals, method = 'CG', tol = 1e-4, jac = backprop, args = (x_vals_modified, y_vals_train, lambda1))#, options = {'disp': True}

# Set up the new theta values we hav from our minimize function
theta_new = minimum.x

final_cost = reg_cost(theta_new, x_vals_modified, y_vals_train, lambda1)
print "Our final cost value is %g." %(final_cost)

# Save the theta values to use later
np.savetxt('finalMNIST500.out', theta_new, delimiter = ',')

