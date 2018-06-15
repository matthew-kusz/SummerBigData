# Import the necessary packages
import struct as st
import gzip
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import grab_data


####### Global variables #######
global_visible_size  = 28 * 28
global_hidden_size = 200

####### Definitions #######
# Reading in MNIST data files	
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

# Import the images we need
theta_final = np.genfromtxt('outputs/finalWeightsRho0.1Lambda0.03Beta0.5Size60000HL200MNIST.out')

# Find the max activations
W1_final, W2_final, b1_final, b2_final = reshape(theta_final)
y = W1_final / np.reshape(np.sqrt(np.sum(W1_final ** 2, axis = 1)), (len(W1_final), 1))

# Now let's show all of the inputs
images1 = []
images6 = []

dim_image = 28
num_row = 14
num_col = 14
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


# Import the file we want
train , labels_train = grab_data.get_data(10, '59')


# Need to know how many inputs we have
m = len(train)

a3_final, a2_final = feedforward(W1_final, W2_final, b1_final, b2_final, train)
'''
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
blackspace = np.ones((8,1))
blackspace2 = np.ones((1,17))
all1 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
for i in range (10):
	temp = np.reshape(patches[i*1000], (8, 8))
	temp2 = np.reshape(a3_final[i*1000], (8, 8))
	all1[i] = np.concatenate((temp, blackspace, temp2), axis = 1)

all_all = np.concatenate((all1[0], blackspace2, all1[1], blackspace2, all1[2], blackspace2, all1[3], blackspace2, all1[4], blackspace2, all1[5], blackspace2, all1[6], blackspace2, all1[7], blackspace2, all1[8], blackspace2, all1[9]), axis = 0)

a = plt.figure(1)
plt.imshow(all_all, cmap = 'binary', interpolation = 'none')
a.show()
'''

raw_input()
