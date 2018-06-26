# Import the necessary packages
import struct as st
import gzip
import numpy as np
import random

####### Global variables #######
global_patch_dim = 15
global_num_patches = 10000
global_image_dim = 28

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

# Generate random patches from the MNIST images
def rand_select(images):
	# Generate a seed so our random values remain the same through each run
	np.random.seed(7)

	images = np.reshape(images, (len(images), global_image_dim, global_image_dim))

	# Pick our random patches from the MNIST dataset images
	patches = np.zeros((global_num_patches, global_patch_dim ** 2)) # (10000, 225)
	for i in range(global_num_patches):
		int_random = random.randint(0, global_image_dim - global_patch_dim)
		pick_image = random.randint(0, len(images) - 1)
		image = images[pick_image, :, :]
		temp = image[int_random: int_random + global_patch_dim, int_random: int_random + global_patch_dim]
		temp = temp.ravel()
		patches[i:i+1, :] = temp

	return patches


####### Code #######
# We need to grab 10000 (15, 15) patches from the MNIST dataset at random to use for our sparse autoencoder

# Extract the MNIST training data sets
x_vals = read_idx('provided_data/train-images-idx3-ubyte.gz', 60000)
x_vals = x_vals / 255.0
x_vals = np.reshape(x_vals, (x_vals.shape[0], (x_vals.shape[1] * x_vals.shape[2])))
print x_vals.shape  # (60000, 784)

# Let's grab the random patches
rand_patches = rand_select(x_vals)

# Flatten completely to save to file
rand_patches = rand_patches.ravel()

# Save the file to use for later
np.savetxt('outputs/MNIST10000patches15x15.out', rand_patches, delimiter = ',')

