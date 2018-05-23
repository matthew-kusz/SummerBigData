# Import the necessary pachages
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random

# Extract the provided data. We need to use scipy since the data is in a matlab file format
data = scipy.io.loadmat('ex3data1.mat')

# The x array is under the name 'X', the y array is under the name 'y'
x_vals = data['X']
y_vals = data['y']

random_images = np.random.randint(0, x_vals.shape[0], 10)
images = x_vals[random_images]

image = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]

for i in range(len(random_images)):
	temp = np.reshape(images[i], (20, 20))
	temp = np.transpose(temp)
	image[i] = temp

total_images = np.concatenate(image, axis = 1)

plt.imshow(total_images, cmap = 'binary')
plt.show()

