# Import the necessary packages
import struct as st
import gzip
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import grab_data


####### Global variables #######
global_visible_size  = 28 * 28
global_hidden_size = 200

####### Definitions #######
# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Feedforward
def feedforward(W1, W2, b1, b2, arr_x):

	'''
	We will be running our sigmoid function twice.
	Tile function allows us to duplicate our rows to the proper dimensions without requiring a for loop.
	This enables each row in our dot product to receive the same bias term. If it were a (25, 10000) array it is equivalent to adding 
	our bias column to each dot product column with just + b1 (since b1 starts as a column).
	'''
	a2 = sigmoid(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))    # (m, 25) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, W2.T) + np.tile(np.ravel(b2), (m, 1)))       # (m, 64) matrix

	return a3, a2

####### Code #######
# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:global_hidden_size * global_visible_size], (global_hidden_size, global_visible_size))
	W2 = np.reshape(theta[global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size], (global_visible_size, global_hidden_size))
	b1 =np.reshape(theta[2 * global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size + global_hidden_size], (global_hidden_size, 1))
	b2 =np.reshape(theta[2 * global_hidden_size * global_visible_size + global_hidden_size: len(theta)], (global_visible_size, 1))
	
	return W1, W2, b1, b2

# Import the images we need
theta_final = np.genfromtxt('outputs/finalWeightsRho0.1Lambda0.03Beta0.5Size60000HL200MNIST.out')

# Find the max activations
W1_final, W2_final, b1_final, b2_final = reshape(theta_final)
y = W1_final / np.reshape(np.sqrt(np.sum(W1_final ** 2, axis = 1)), (len(W1_final), 1))

# Now let's show all of the inputs
images1 = []
images6 = []

dim_image = 28

# Set up the number of columns and rows we want in our grid
num_row = 14
num_col = 14

# Dividers
black_space = np.ones((dim_image, 1)) * y.max()
black_space2 = np.ones((1, dim_image * num_col + num_col - 1)) * y.max()

for i in range(num_row):
	for j in range(num_col):
		if (j == 0):
			images1 = np.reshape(y[j + i * num_col], (dim_image, dim_image))
		else:
			temp = np.reshape(y[j + i * num_col], (dim_image, dim_image))
			images1 = np.concatenate((images1, black_space, temp), axis = 1)
			
	if (i == 0):
		images6 = images1
	else:
		images6 = np.concatenate((images6, black_space2, images1), axis = 0)

d = plt.figure(2)
plt.imshow(images6, cmap = 'binary', interpolation = 'none')
d.show()

raw_input()
