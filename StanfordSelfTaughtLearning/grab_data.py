# Import the necessary packages
import struct as st
import gzip
import numpy as np

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

####### Code #######
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

# Set up the arrays we need to store the data after we sort it
train = []
test = []

# We need numbers 5-9 for our training set and 0-4 for out test set
for i in range(size):
	if (y_vals[i] == 5 or y_vals[i] == 6 or y_vals[i] == 7 or y_vals[i] == 8 or y_vals[i] == 9):
		if (len(train) == 0):
			train = np.reshape(x_vals[i], (1, len(x_vals[i])))
			# y_train = np.reshape(y_vals[i], (1, len(y_vals[i])))
		else:
			train = np.concatenate((train, np.reshape(x_vals[i], (1, len(x_vals[i])))), axis = 0)
			# y_train = np.concatenate((y_train, np.reshape(y_vals[i], (1, len(y_vals[i])))), axis = 0)
		if (len(test) == 0):
			test = np.reshape(x_vals[i], (1, len(x_vals[i])))
			# y_test = np.reshape(y_vals[i], (1, len(y_vals[i])))
		else:
			test = np.concatenate((test, np.reshape(x_vals[i], (1, len(x_vals[i])))), axis = 0)
			# y_test = np.concatenate((y_test, np.reshape(y_vals[i], (1, len(y_vals[i])))), axis = 0)

print train.shape
print test.shape


train = train.ravel()
test = test.ravel()
# Save the trains data to use later
np.savetxt('data/trainDatafrom60000', train, delimiter = ',')

# Save the tests data to use later
np.savetxt('data/trainDatafrom60000', test, delimiter = ',')
