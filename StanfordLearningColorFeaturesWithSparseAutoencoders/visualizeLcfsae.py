# Import the necessary packages
import struct as st
import gzip
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math


####### Global variables #######
global_image_channels = 3
global_patch_dim = 8
global_visible_size = global_patch_dim * global_patch_dim * global_image_channels
global_hidden_size = 400
global_epsilon = 0.1           # ZCA whitening

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
	a2 = sigmoid(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))    # (m, 400) matrix

	# Second run
	a3 = np.dot(a2, W2.T) + np.tile(np.ravel(b2), (m, 1))       # (m, 192) matrix

	return a3, a2

# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:global_hidden_size * global_visible_size], (global_hidden_size, global_visible_size))
	W2 = np.reshape(theta[global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size], (global_visible_size, global_hidden_size))
	b1 =np.reshape(theta[2 * global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size + global_hidden_size], (global_hidden_size, 1))
	b2 =np.reshape(theta[2 * global_hidden_size * global_visible_size + global_hidden_size: len(theta)], (global_visible_size, 1))
	
	return W1, W2, b1, b2

def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0.0
	nMax = 1.0
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin

####### Comparing our inputs with our outputs #######
# Import the file we want
data = scipy.io.loadmat('data/stlSampledPatches.mat')
patches = data['patches'].T                # (m, 192)

# Need to know how many inputs we have
m = len(patches)
n = len(patches[0])

# Import the weights we need
theta_final = np.genfromtxt('outputs/finalWeightsRho0.035Lambda0.003Beta5.0Size100000HL400.out')
W1_final, W2_final, b1_final, b2_final = reshape(theta_final)

# Forward propagate
a3_final, a2_final = feedforward(W1_final, W2_final, b1_final, b2_final, patches)

# Normalize our inputs and outputs so each row ranges from 0-1
for i in range(m):
	a3_final[i] = Norm(a3_final[i])

for i in range(m):
	patches[i] = Norm(patches[i])

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
black_space = np.zeros((global_patch_dim,blackbar_length,global_image_channels))
black_space2 = np.zeros((blackbar_length, global_patch_dim * num_col + num_col * blackbar_length - blackbar_length, global_image_channels))

# Setting up our grid
for i in range(num_row):
	for j in range(num_col):
		if (j == 0):
			comparison1 = np.reshape(patches[i*2000], (global_patch_dim, global_patch_dim, global_image_channels))
		else:
			temp = np.reshape(a3_final[i*2000], (global_patch_dim, global_patch_dim, global_image_channels))
			comparison1 = np.concatenate((comparison1, black_space, temp), axis = 1)

	if (i == 0):
		comparison2 = comparison1
	else:
		comparison2 = np.concatenate((comparison2, black_space2, comparison1), axis = 0)

# Displaying the grid			
a = plt.figure(1)
plt.imshow(comparison2, interpolation = 'nearest')
a.show()

####### Visualizing our max activations #######
# We need to reuse our whitening variable to apply to our weights
ZCA_whitening = np.genfromtxt('outputs/ZCAwhitening0.035Lambda0.003Beta5.0Size100000HL400.out')
ZCA_whitening = np.reshape(ZCA_whitening, (192, 192))

# Find the max activations
y = W1_final / np.reshape(np.sqrt(np.sum(W1_final ** 2, axis = 1)), (len(W1_final), 1))
y = np.dot(y, ZCA_whitening)

# Normalize our matrix
y = (y + 1.0) / 2.0

# Now let's show all of the inputs
images1 = []
images2 = []

# Set up the number of columns and rows that we want in our grid
num_row = 20
num_col = 20

# Dividers
blackbar_length = 2
black_space = np.ones((global_patch_dim, blackbar_length, global_image_channels)) * np.max(y)
black_space2 = np.ones((blackbar_length, global_patch_dim * num_col + num_col * blackbar_length - blackbar_length, global_image_channels)) * np.amax(y)

# Setting up our grid
for i in range(num_row):
	for j in range(num_col):
		if (j == 0):
			# First set up our image in red, then blue, then green
			s = global_patch_dim**2
    			img = np.zeros((global_patch_dim, global_patch_dim, 3))
    			img[:,:,0] = y[j + i * num_col][:s].reshape(global_patch_dim, global_patch_dim)
   			img[:,:,1] = y[j + i * num_col][s:2*s].reshape(global_patch_dim, global_patch_dim)
    			img[:,:,2] = y[j + i * num_col][2*s:].reshape(global_patch_dim, global_patch_dim)
			images1 = img

		else:
			s = global_patch_dim**2
    			img = np.zeros((global_patch_dim, global_patch_dim, 3))
    			img[:,:,0] = y[j + i * num_col][:s].reshape(global_patch_dim, global_patch_dim)
   			img[:,:,1] = y[j + i * num_col][s:2*s].reshape(global_patch_dim, global_patch_dim)
    			img[:,:,2] = y[j + i * num_col][2*s:].reshape(global_patch_dim, global_patch_dim)
			images1 = np.concatenate((images1, black_space, img), axis = 1)
			
	if (i == 0):
		images2 = images1
	else:
		images2 = np.concatenate((images2, black_space2, images1), axis = 0)

# Displaying the grid
d = plt.figure(2)
plt.imshow(images2, interpolation = 'nearest')
d.show()

# Allows us to view all images at once
raw_input()
