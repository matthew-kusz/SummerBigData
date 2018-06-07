# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random

####### Generate training set #######

# Extract the provided data. We need to use scipy since the data is in a matlab file format
images_data = scipy.io.loadmat('starter/IMAGES.mat')

# The images that we need to extract are under the name 'IMAGES'
images = images_data['IMAGES']

patch_size = 8       # We want to use 8x8 patches
num_patches = 100    # Total number of patches we will have

# Set up an array of zeros for the patches (64, 10000)
patches = np.zeros((patch_size ** 2, num_patches))

# Now we want to break the image up into patches
for i in range(len(patches[0])):
	int_random = random.randint(0, 504)
	pick_image = random.randint(0, 9)
	image = images[:,:, pick_image]
	temp = image[int_random: int_random + 8, int_random: int_random + 8]
	temp = np.reshape(temp, (64, 1))
	patches[:, i:i+1] = temp

'''
# Check to make sure our code is running correctly
image2 = np.reshape(patches[:,0:1], (8, 8))
plt.imshow(image2, cmap = 'binary', interpolation = 'none')
plt.show()
'''

# Save our patches array to a file to be used later
patches = np.ravel(patches)
np.savetxt('outputs/100Random8x8.out', patches, delimiter = ',')
