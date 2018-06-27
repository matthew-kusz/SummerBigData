# Import the necessary packages
import struct as st
import gzip
import numpy as np
# from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

####### Global variables #######
global_num_train = 60000
global_input_layer = 784
global_hidden_layer1 = 350
global_output_layer = 10
global_epochs = 1
global_batch_size = 86
global_num_classes = 10

# So our random values stay the same for each run
np.random.seed(7)

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

# Extract the MNIST training data sets
def get_train(size):
	x_vals = read_idx('data/train-images-idx3-ubyte.gz', size)
	x_vals = x_vals / 255.0
	x_vals = np.reshape(x_vals, (x_vals.shape[0], (x_vals.shape[1] * x_vals.shape[2])))
	y_vals = read_idx('data/train-labels-idx1-ubyte.gz', size)
	y_vals = np.reshape(y_vals, (len(y_vals), 1))
	print x_vals.shape, y_vals.shape
	
	return x_vals, y_vals

####### Code #######
# Extract the MNIST training data sets
train_images, train_labels = get_train(global_num_train)
m = len(train_images)

# Set up a matrix that will be either 1 or 0 depending on what we are looking at
# Y_train = to_categorical(Y_train, num_classes = 10)
y_vals_train = np.zeros((len(train_labels), global_num_classes))
for i in range(global_num_classes):
	# Set up an array with the values that stand for each label
	arr_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	
	for j in range(len(train_labels)):
		if (train_labels[j] == arr_num[i]):
			y_vals_train[j, i:] = 1
		
		else:
			y_vals_train[j, i:] = 0

# Setting up the Keras neural network
# Create our model (currently has 1 hidden layer and using softmax regression)
model = Sequential()
'''
model.add(Dense(global_hidden_layer1, input_dim = global_input_layer, activation = 'relu'))
model.add(Dense(global_output_layer, activation = 'softmax'))
'''
#NOT MY CODE
train_images = np.reshape(train_images, (m, 28, 28, 1))
np.random.seed(2)

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# Compile our model
model.compile(optimizer = 'SGD', loss='categorical_crossentropy', metrics=['accuracy'])
print model.summary()

# Fit our model
model.fit(train_images, y_vals_train, epochs= global_epochs, batch_size = global_batch_size)

# Evaluate our model
scores = model.evaluate(train_images, y_vals_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Save our model and weights
#model.save('models/ModelDataSize'+str(m)+'Epoch'+ str(global_epochs)+'BatchSize'+ str(global_batch_size))
#model.save_weights('weights/WeightsDataSize'+str(m)+'Epoch'+ str(global_epochs)+'BatchSize'+ str(global_batch_size))
