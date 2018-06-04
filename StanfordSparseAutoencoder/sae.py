# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random

####### Definitions #######
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

####### Generate training set #######

# Extract the provided data. We need to use scipy since the data is in a matlab file format
images_data = scipy.io.loadmat('starter/IMAGES.mat')

# The images that we need to extract are under the name 'IMAGES'
images = images_data['IMAGES']
print images.shape

patch_size = 8       # We want to use 8x8 patches
num_patches = 10000  # Total number of patches we will have

# Set up an array of zeros for the patches (64, 10000)
patches = np.zeros((patch_size ** 2, num_patches))

# Let's look at one of the images
pick_image = 0
pick_image = int(input('Enter digit representing an image (0-9):'))
while (pick_image > 9 or pick_image < 0):
	pick_image = int(input('Please pick a digit in the range 0-9:'))

image = images[:,:, pick_image]
plt.imshow(image, cmap = 'binary')
# plt.show()

# Now we want to break the image up into patches
for i in range(len(patches[0])):
	int_random = random.randint(0, 504)
	temp = image[int_random: int_random + 8, int_random: int_random + 8]
	temp = np.reshape(temp, (64, 1))
	patches[:, i:i+1] = temp

# Check to make sure our code is running correctly
image2 = np.reshape(patches[:,0:1], (8, 8))
plt.imshow(image2, cmap = 'binary', interpolation = 'none')
# plt.show()

####### Sparse autoencoder objective #######
