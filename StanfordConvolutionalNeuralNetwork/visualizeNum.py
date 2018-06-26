# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

####### Global variables #######
global_patch_dim = 15
global_visible_size = global_patch_dim ** 2
global_hidden_size = 100

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
	a2 = sigmoid(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))    # (10000, 25) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, W2.T) + np.tile(np.ravel(b2), (m, 1)))       # (10000, 64) matrix

	return a3, a2

####### Code #######
# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:global_hidden_size * global_visible_size], (global_hidden_size, global_visible_size))
	W2 = np.reshape(theta[global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size], (global_visible_size, global_hidden_size))
	b1 =np.reshape(theta[2 * global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size + global_hidden_size], (global_hidden_size, 1))
	b2 =np.reshape(theta[2 * global_hidden_size * global_visible_size + global_hidden_size: len(theta)], (global_visible_size, 1))
	
	return W1, W2, b1, b2

# Import the data we need
patches = np.genfromtxt('outputs/MNIST10000patches15x15.out')  # (10000, 225)
patches = np.reshape(patches, (len(patches) / global_visible_size, global_visible_size))
m = len(patches)
n = len(patches[0])

# We need our values in patches and a3_final to range from 0 to 1
old_min = -1
old_max = 1
new_min = 0
new_max = 1
patches = ((patches - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

# Import the weights we need
theta_final = np.genfromtxt('outputs/finalWeightsMNISTSize10000Patches15x15L0.0001B0.5Rho0.01HL100.out')

W1_final, W2_final, b1_final, b2_final = reshape(theta_final)
a3_final, a2_final = feedforward(W1_final, W2_final, b1_final, b2_final, patches)

for i in range(m):
	a3_final[i] = ((a3_final[i] - a3_final[i].min()) / (a3_final[i].max() - a3_final[i].min())) * (new_max - new_min) + new_min

for i in range(m):
	patches[i] = ((patches[i] - patches[i].min()) / (patches[i].max() - patches[i].min())) * (new_max - new_min) + new_min

# Let's see how accurate we were
error = np.zeros((m, n))
for i in range(m):
	error[i] = np.abs(patches[i] - a3_final[i])

print "The average difference between the output pixel and the input pixel is %g." %(np.mean(error))

# Plot an image of a1 and a3 side by side to see how accurate the output is to the original
comparison1 = []
comparison2 = []

# Set up the number of columns and rows that we want in our grid
num_row = 10
num_col = 2

# Dividers
blackbar_length = 2
black_space = np.zeros((global_patch_dim, blackbar_length))
black_space2 = np.zeros((blackbar_length, global_patch_dim * num_col + num_col * blackbar_length - blackbar_length))

# Setting up our grid
for i in range(num_row):
	for j in range(num_col):
		if (j == 0):
			comparison1 = np.reshape(patches[i*1000], (global_patch_dim, global_patch_dim))
		else:
			temp = np.reshape(a3_final[i*1000], (global_patch_dim, global_patch_dim))
			comparison1 = np.concatenate((comparison1, black_space, temp), axis = 1)

	if (i == 0):
		comparison2 = comparison1
	else:
		comparison2 = np.concatenate((comparison2, black_space2, comparison1), axis = 0)

# Displaying the grid			
a = plt.figure(1)
plt.imshow(comparison2, cmap = 'binary', interpolation = 'none')
a.show()

####### Visualizing our max activations #######
# Find the max activations
y = W1_final / np.reshape(np.sqrt(np.sum(W1_final ** 2, axis = 1)), (len(W1_final), 1))

# Now let's show all of the inputs
images1 = []
images2 = []

# Set up the number of columns and rows that we want in our grid
num_row = 10
num_col = 10

# Dividers
blackbar_length = 2
black_space = np.ones((global_patch_dim, blackbar_length)) * np.max(y)
black_space2 = np.ones((blackbar_length, global_patch_dim * num_col + num_col * blackbar_length - blackbar_length)) * np.amax(y)

# Setting up our grid
for i in range(num_row):
	for j in range(num_col):
		if (j == 0):
			images1 = np.reshape(y[j + i * num_col], (global_patch_dim, global_patch_dim))

		else:
			temp = np.reshape(y[j + i * num_col], (global_patch_dim, global_patch_dim))
			images1 = np.concatenate((images1, black_space, temp), axis = 1)
			
	if (i == 0):
		images2 = images1
	else:
		images2 = np.concatenate((images2, black_space2, images1), axis = 0)

# Displaying the grid
d = plt.figure(2)
plt.imshow(images2, cmap = 'binary', interpolation = 'none')
d.show()

# Allows us to view all images at once
raw_input()
