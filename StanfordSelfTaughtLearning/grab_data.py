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

def get_data(size, string):
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
	labels_test = []
	labels_train = []

	# We need numbers 5-9 for our training set and 0-4 for out test set
	for i in range(size):
		if (y_vals[i] == 5 or y_vals[i] == 6 or y_vals[i] == 7 or y_vals[i] == 8 or y_vals[i] == 9):
			if (len(train) == 0):
				train = np.reshape(x_vals[i], (1, len(x_vals[i])))
				#labels_train = np.reshape(y_vals[i], (1, len(y_vals[i])))
			else:
				train = np.concatenate((train, np.reshape(x_vals[i], (1, len(x_vals[i])))), axis = 0)
				#labels_train = np.concatenate((labels_train, np.reshape(y_vals[i], (1, len(y_vals[i])))), axis = 0)
		'''
		else:
			if (len(test) == 0):
				test = np.reshape(x_vals[i], (1, len(x_vals[i])))
				labels_test = np.reshape(y_vals[i], (1, len(y_vals[i])))
			else:
				test = np.concatenate((test, np.reshape(x_vals[i], (1, len(x_vals[i])))), axis = 0)
				labels_test = np.concatenate((labels_test, np.reshape(y_vals[i], (1, len(y_vals[i])))), axis = 0)
		'''

	if (string == '59'):
		print "59"
		print train.shape
		# print labels_train.shape
		return train, labels_train
	
	elif (string == '04'):
		print "04"
		print test.shape
		print labels_test.shape
		return test, labels_test
