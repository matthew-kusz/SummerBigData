# Import the necessary pachages
import struct as st
import gzip
import scipy.io
import scipy.optimize
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

# Reading in MNIST data files	
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr

# Set up how large we want our data set (max of 60,000)
size = 60000

# Extract the MNIST training data sets
x_vals = read_idx('data/train-images-idx3-ubyte.gz', size)
x_vals = x_vals / 255.0
x_vals = np.reshape(x_vals, (x_vals.shape[0], (x_vals.shape[1] * x_vals.shape[2])))
y_vals = read_idx('data/train-labels-idx1-ubyte.gz', size)
y_vals = np.reshape(y_vals, (len(y_vals), 1))
print x_vals.shape
print y_vals.shape

# Set up a an array of random 10 images [a (10, 400) vector]
random_images = np.random.randint(0, x_vals.shape[0] - 1, 10)
images = x_vals[random_images]
print images.shape

# Now reshape images back into (20, 20) matrices and transpose to flip images right-side up
image = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
image2 = image

# Rotating the image
for i in range(len(random_images)):
	temp = np.reshape(images[i], (28, 28))
	print temp.shape
	temp = scipy.ndimage.rotate(temp, 45, reshape = False)
	print temp.shape
	image[i] = temp

# Put all the images together
total_images = np.concatenate(image, axis = 1)

# Plot the results
plt.imshow(total_images, cmap = 'binary')
plt.show()
