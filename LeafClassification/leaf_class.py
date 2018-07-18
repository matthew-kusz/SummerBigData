# Import the necessary packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd                         	 # For reading in and writing files
from keras.models import Sequential, Model       # Linearly sets up model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, concatenate, Input
import matplotlib.image as mpimg           	 # Reading images to numpy arrays
import scipy.ndimage as ndi            	   	 # Finding the center of the leaves
from sklearn.preprocessing import LabelEncoder   # Preprocessing
from sklearn.preprocessing import StandardScaler # Preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from keras.utils import np_utils, plot_model     # Used to set up one-hot scheme, visualizing my model
from keras.callbacks import EarlyStopping 	 # Used to prevent overfitting
from keras.callbacks import ModelCheckpoint      # Gives us the best weights obtained during fitting
from keras.optimizers import SGD
import argparse
from scipy.misc import imresize
import visualize
from sklearn.decomposition import PCA

####### Global Variables #######
parser = argparse.ArgumentParser()
parser.add_argument('disp_stats', 
	help = 'Use 1/0 (True/False) to indicate if you want to display model stats or not.', type = int)
parser.add_argument('save_stats', 
	help = 'Use 1/0 (True/False) to indicate if you want to save model stats or not.', type = int)
parser.add_argument('save_prob', 
	help = 'Use 1/0 (True/False) to indicate if you want to save probabilities or not.', type = int)
parser.add_argument('leaf_stats', 
	help = 'Use 1/0 (True/False) to indicate if you want to display leaf images and stats or not.', type = int)
parser.add_argument('global_max_epochs', help = 'Max amount of epochs allowed.', type = int)
parser.add_argument('global_batch_size', help = 'Numer of samples per gradient update.', type = int)

args = parser.parse_args()

global_num_train = 990
global_num_test = 594
global_total_img = global_num_train + global_num_test
global_input_layer = 192
global_hidden_layers = [400, 280, 140]
global_output_layer = 99
global_num_classes = 99
global_max_dim = 50

path = 'data_provided/unzip_images'
filename1 = 'graphs/lossEpoch' + str(args.global_max_epochs) + 'Batch' + str(args.global_batch_size) + 'Softmax.png'
filename2 = 'graphs/accEpoch' + str(args.global_max_epochs) + 'Batch' + str(args.global_batch_size) + 'Softmax.png'
filename3 = 'submissions/submissionEpoch' + str(args.global_max_epochs) + 'Batch' + str(args.global_batch_size) + 'Softmax.csv'

# Set up a seed so that our results don't fluctuate over multiple tests
np.random.seed(1)

####### Definitions #######
# Set up all of the images in a matrix
def grab_images(tr_ids, te_ids):
	# Full set
	matrix = np.zeros((2, global_total_img))
	images_list = []
	for i in range(global_total_img):
		img = mpimg.imread('data_provided/unzip_images/images/' + str(i + 1) + '.jpg')
		matrix[:,i] = np.shape(img)
		images_list.append(img)

	# We will want to learn features on the images that belong to our training set	
	train_list = []
	for i, img in enumerate(tr_ids):
		train_list.append(images_list[img - 1])
	
	# We might want to look at some test images too	
	test_list = []
	for i, img in enumerate(te_ids):
		test_list.append(images_list[img - 1])

	return images_list, train_list, test_list
'''
# Model set-up for CNN
def create_model_conv():

	mod2 = Sequential()
	mod2.add(Conv2D(input_shape = (50, 50, 1), filters = 8, kernel_size = (5, 5), activation ='relu', padding = 'Same'))
	mod2.add(MaxPool2D(pool_size = (25, 25)))
	mod2.add(Dropout(0.2))
	mod2.add(Flatten())
	mod2.add(Dense(192, activation = "relu"))
	mod2.add(Dropout(0.2))
	# mod2.add(Dense(99, activation = "softmax"))
	
	return mod2
'''
# Model set-up for softmax regression
def create_model_softmax():
		
	mod1 = Sequential()	
	
	mod1.add(Dense(global_hidden_layers[0], input_dim = global_input_layer, activation = 'relu'))
	mod1.add(Dropout(0.3))
	
	if (len(global_hidden_layers) > 1):
		for i in range(len(global_hidden_layers) - 1):
			mod1.add(Dense(global_hidden_layers[i + 1], activation = 'relu'))
			mod1.add(Dropout(0.2))	
	
	mod1.add(Dense(global_output_layer, activation = 'softmax'))

	
	return mod1

def combined_model():

	first_input = Input(shape=(global_max_dim, global_max_dim, 1))
	x = Conv2D(filters = 8, kernel_size = (5, 5), activation ='relu', padding = 'Same')(first_input)
	x = MaxPool2D(pool_size = (25, 25))(x)
	x = Dropout(0.2)(x)
	x = Flatten()(x)
	x = Dense(192, activation = "relu")(x)
	final_first_input_layer = Dropout(0.2)(x)

	second_input = Input(shape=(global_input_layer, ))

	merge_one = concatenate([final_first_input_layer, second_input])

	x = Dense(global_hidden_layers[0], activation = 'relu')(merge_one)
	x = Dropout(0.2)(x)
	
	if (len(global_hidden_layers) > 1):
		for i in range(len(global_hidden_layers) - 1):
			x = Dense(global_hidden_layers[i + 1], activation = 'relu')(x)
			x = Dropout(0.2)(x)
	
	final = Dense(global_output_layer, activation = 'softmax')(x)
	model = Model(inputs=([first_input, second_input]), outputs=final)

	return model

################################################
# From https://www.kaggle.com/abhmul/keras-convnet-lb-0-0052-w-visualization/notebook
################################################
"""
Resize the image to so the maximum side is of size max_dim
Returns a new image of the right size
"""
'''
def resize_img(img, max_dim):
    	# Get the axis with the larger dimension
    	max_ax = max((0, 1), key=lambda i: img.size[i])

    	# Scale both axes so the image's largest dimension is max_dim
    	scale = max_dim / float(img.size[max_ax])

    	return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
'''
"""
Takes as input an array of image ids and loads the images as numpy
arrays with the images resized so the longest side is max-dim length.
If center is True, then will place the image in the center of
the output array, otherwise it will be placed at the top-left corner.
"""
'''
def load_image_data(ids, max_dim=165, center=True):
    	# Initialize the output array
   	X = np.empty((len(ids), max_dim, max_dim, 1))

   	for i, idee in enumerate(ids):
        	# Turn the image into an array
        	x = resize_img(load_img(os.path.join(path, 'images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
        	x = img_to_array(x)

		print x.shape
		d = plt.figure(1)
		plt.imshow(x[:,:,0], cmap = 'binary', interpolation = 'none')
		d.show()

        	# Get the corners of the bounding box for the image
        	length = x.shape[0]
        	width = x.shape[1]
        	if center:
            		h1 = int((max_dim - length) / 2)
            		h2 = h1 + length
            		w1 = int((max_dim - width) / 2)
            		w2 = w1 + width
        	else:
            		h1, w1 = 0, 0
            		h2, w2 = (length, width)
        	
		# Insert into image matrix
        	X[i, h1:h2, w1:w2, 0:1] = x
		print X[i,:,:,0].shape
		c = plt.figure(2)
		plt.imshow(X[i,:,:,0], cmap = 'binary', interpolation = 'none')
		c.show()
		#raw_input()
		break

    	# Scale the array values so they are between 0 and 1
    	return np.around(X / 255.0)
'''
################################################
################################################
"""
Resize the image to so the maximum side is of size max_dim
Returns a new image of the right size
"""

def resize_img(img, max_dim):
    	# Get the axis with the larger dimension
    	max_ax = max((0, 1), key=lambda i: img.shape[i])
    	# Scale both axes so the image's largest dimension is max_dim
   	scale = max_dim / float(img.shape[max_ax])
    	return np.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)))

def reshape_img(images, max_dim, center = True):
	
	modified = np.zeros((len(images), max_dim, max_dim, 1))
	for i in range(len(images)):
		temp = resize_img(images[i], max_dim = max_dim)
		x = imresize(images[i], (temp.shape[0], temp.shape[1]), interp = 'nearest').reshape(temp.shape[0], 
			temp.shape[1], 1)

		length = x.shape[0]
		width = x.shape[1]
		if center:
			h1 = int((max_dim - length) / 2)
           	 	h2 = h1 + length
           		w1 = int((max_dim - width) / 2)
          		w2 = w1 + width
       		else:
           		h1, w1 = 0, 0
          		h2, w2 = (length, width)
	
		modified[i, h1:h2, w1:w2, 0:1] = x

	return  np.around(modified / 255.0)

####### Code #######
# We need to extract the data given
# Set up our training data
train = pd.read_csv('data_provided/train.csv')

# Extract the species of each leaf
y_raw = train.pop('species')

# Label each species from 0 - n-1
le = LabelEncoder()
# fit() calculates the mean and std, transform() centers and scales data
y = le.fit(y_raw).transform(y_raw)
			
# Grab the classes (will be used to set up our submition file)
classes = le.classes_
# Setting up one-hot scheme
y_train = np_utils.to_categorical(y)

# Extract the id of each leaf
train_ids = train.pop('id')

# Set up our testing data
test = pd.read_csv('data_provided/test.csv')

# Extract the id of each leaf
test_ids = test.pop('id')
'''
# fit_transform() calculates the mean and std and also centers and scales data
x_train = StandardScaler().fit_transform(train)
x_test = StandardScaler().fit_transform(test)
'''
# Load up all of the images
print 'Loading images...'
img_list, train_list, test_list = grab_images(train_ids, test_ids)
print 'Finished.'
	
# We need to reshape our images so they are all the same dimensions
train_mod_list = reshape_img(train_list, global_max_dim)
test_mod_list = reshape_img(test_list, global_max_dim)
'''
# Let's try to get some engineered features
train_width = np.zeros((len(train_mod_list), 1)) 
train_height = np.zeros((len(train_mod_list), 1))
train_asp_ratio = np.zeros((len(train_mod_list), 1))
train_square = np.zeros((len(train_mod_list), 1))
test_width = np.zeros((len(test_mod_list), 1))
test_height = np.zeros((len(test_mod_list), 1))
test_asp_ratio = np.zeros((len(test_mod_list), 1))
test_square = np.zeros((len(test_mod_list), 1))

for i in range(len(train_mod_list)):
	train_width[i] = train_list[i].shape[1]
	train_height[i] = train_list[i].shape[0]
	train_asp_ratio[i] = train_list[i].shape[1] / train_list[i].shape[0]
	train_square[i] = train_list[i].shape[1] * train_list[i].shape[0]

for i in range(len(test_mod_list)):
	test_width[i] = test_list[i].shape[1]
	test_height[i] = test_list[i].shape[0]
	test_asp_ratio[i] = test_list[i].shape[1] / test_list[i].shape[0]
	test_square[i] = test_list[i].shape[1] * test_list[i].shape[0]

train = np.concatenate((train, train_width, train_height, train_asp_ratio, train_square), axis = 1)
test = np.concatenate((test, test_width, test_height, test_asp_ratio, test_square), axis = 1)
global_input_layer += 4
'''
'''
# Let's apply PCA to the images and then add them to our pre-extracted features and see what we get
pca = PCA(0.80)
pca.fit(train_mod_list.reshape(len(train_mod_list), 50 * 50))
global_input_layer += pca.n_components_
print pca.n_components_
train_PCA = pca.transform(train_mod_list.reshape(len(train_mod_list), 50 * 50))
test_PCA = pca.transform(test_mod_list.reshape(len(test_mod_list), 50 * 50))

train = np.concatenate((train, train_PCA), axis = 1)
test = np.concatenate((test, test_PCA), axis = 1)
'''
x_train = StandardScaler().fit_transform(train)
x_test = StandardScaler().fit_transform(test)

# Look at images and some stats of leaves
if args.leaf_stats:
	visualize.image_similarity(train_mod_list, y, classes)
	visualize.visualize_leaves(train_mod_list, y, classes, show1 = False, show2 = True)

# test_mod_list = reshape_img(test_list)
# visualize.visualize_leaves(test_mod_list, y, classes, show1 = True, show2 = False)
# Setting up the Keras neural network
# model = create_model_softmax()
# model = create_model_conv()
model = combined_model()

# Compile our model
sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
print model.summary()
plot_model(model, to_file = 'modelCombined.png', show_shapes = True)

'''
Fit our model
Early stopping helps prevent over fitting by stopping our fitting function 
if our val_loss doesn't decrease after a certain number of epochs (called patience)
Model checkpoint saves the best weights obtained during training
'''

early_stopper= EarlyStopping(monitor = 'val_loss', patience = 300, verbose = 1, mode = 'auto')
model_checkpoint = ModelCheckpoint('bestWeights2.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True)
history = model.fit(([train_mod_list, x_train]), y_train, epochs = args.global_max_epochs, batch_size = args.global_batch_size,
	verbose = 0, validation_split = 0.1, shuffle = True, callbacks = [early_stopper, model_checkpoint])

# Check Keras' statistics
if args.disp_stats:
	print 'Displaying stats...'
	visualize.plt_perf(history, filename1, filename2, p_loss = True, p_acc = True, val = True, save = args.save_stats)

# Reload our best weights
model.load_weights('bestWeights2.hdf5')

# Test our model on the test set
y_pred = model.predict_proba(x_test)
print '\n'
'''
indices = np.zeros((len(y_pred), 1))
for i in range(len(y_pred)):
	indices[i] = np.argmax(y_pred[i])

for i in range(len(classes)):
	temp = np.where(indices == i)

	if len(temp[0]) != 6:
		print i, len(temp[0])

visualize.visualize_leaves(test_mod_list, indices, classes, show1 = False, show2 = True)
'''
'''
Let's see how accurace our model is
WARNING
This is based solely on low probability for guess = incorrect
Has been tested and shows correct guesses majority of the time (bad)
'''
'''
x = []
x_val = []
total = 0
for i in range(len(y_pred)):
	if (np.amax(y_pred[i]) < 0.5):
		x = np.append(x, np.argmax(y_pred[i]))
		x_val = np.append(x_val, test_ids[i])
		total += 1
print 'Total number of < 0.5 predictions: %g' %(total)

x2 = np.ones(global_num_classes)
for i in range (len(x2)):
	x2[i] = i
for i in range(len(x)):
	for j in range(len(x2)):
		if (x2[j] == x[i]):	
			print classes[j]
			print int(x_val[i])
			img = mpimg.imread('data_provided/unzip_images/images/' + str(int(x_val[i])) + '.jpg')
			plt.title(classes[j])
			plt.imshow(img, cmap = 'binary')
			plt.show()
'''

# Set up the predictions into the correct format to submit to Kaggle
y_pred = pd.DataFrame(y_pred, index = test_ids, columns = classes)

if args.save_prob:
	print 'Saving to file...'
	fp = open(filename3,'w')
	fp.write(y_pred.to_csv())

print 'Finished.'

