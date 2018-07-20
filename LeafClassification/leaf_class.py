# Import the necessary packages
# Misc
import numpy as np				 	 # Allow for easier use of arrays and linear algebra
import matplotlib                                	 # Imported for use of matplotlib.use('Agg') 
#matplotlib.use('Agg')                            	 # Use if submitting job to the supercomputer
import matplotlib.pyplot as plt                  	 # For plotting
import pandas as pd                         	 	 # For reading in and writing files
import matplotlib.image as mpimg           	 	 # Reading images to numpy arrays
import scipy.ndimage as ndi            	   	 	 # Finding the center of the leaves
import argparse                                  	 # For inputing values outside of the code
from scipy.misc import imresize                  	 # For resizing the images
import visualize                                 	 # Python code for visualizing images

# sklearn
from sklearn.preprocessing import LabelEncoder  	 # Preprocessing
from sklearn.preprocessing import StandardScaler 	 # Preprocessing
from sklearn.decomposition import PCA           	 # Preprocessing

# Keras
from keras.models import Sequential, Model      	 # Linearly sets up model, allows the use of keras' API
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, concatenate, Input
from keras.utils import np_utils, plot_model             # Used to set up one-hot scheme, visualizing model
from keras.callbacks import EarlyStopping 	         # Used to prevent overfitting
from keras.callbacks import ModelCheckpoint              # Gives us the best weights obtained during fitting
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img
 # Generate more images
from keras.optimizers import SGD                         # Enable the modification of optimizer SGD

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
global_max_dim = 100

filename1 = 'graphs/lossEpoch' + str(args.global_max_epochs) + 'Batch' + str(args.global_batch_size) + 'Softmax.png'
filename2 = 'graphs/accEpoch' + str(args.global_max_epochs) + 'Batch' + str(args.global_batch_size) + 'Softmax.png'
filename3 = 'submissions/submissionEpoch' + str(args.global_max_epochs) + 'Batch' + str(args.global_batch_size) + 'Softmax.csv'

# Set up a seed so that our results don't fluctuate over multiple tests
np.random.seed(1)

####### Definitions #######
class ImageDataGenerator2(ImageDataGenerator):
    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator2(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class NumpyArrayIterator2(NumpyArrayIterator):
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            # We changed index_array to self.index_array
            self.index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(self.index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y

def grab_images(tr_ids, te_ids):
	'''
	Reads in image files to save to a list, then separates the pictures between training and testing ids

	Parameters:
	tr_ids - training image ids
	te_ids - testing image ids

	Returns:
	images_list - full list of images
	train_list - list only consisting of training images
	test_list - list only consisting of testing images
	'''	

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

def create_model_softmax():
	'''
	This model uses only the pre-extracted features of the leaves to create a softmax regression model
	Returns the built model
	'''
		
	mod1 = Sequential()	
	
	mod1.add(Dense(global_hidden_layers[0], input_dim = global_input_layer, init='uniform', activation = 'relu'))
	mod1.add(Dropout(0.3))
	
	# For loop allows for easier adding and removing of hidden layers adjusting their sizes
	if (len(global_hidden_layers) > 1):
		for i in range(len(global_hidden_layers) - 1):
			mod1.add(Dense(global_hidden_layers[i + 1], activation = 'relu'))
			mod1.add(Dropout(0.2))	
	
	mod1.add(Dense(global_output_layer, activation = 'softmax'))
	
	return mod1

def create_model_combined():
	'''
	This model uses both the images and the and pre-extracted features of the leaves to create a CNN model
	Returns the built model
	'''
	# Obtaining features from the images
	first_input = Input(shape=(global_max_dim, global_max_dim, 1))
	x = Conv2D(filters = 32, kernel_size = (25, 25), activation ='relu', padding = 'Same', strides = 2)(first_input)
	x = MaxPool2D(pool_size = (10, 10))(x)
	x = Dropout(0.2)(x)
	x = Conv2D(filters = 8, kernel_size = (5, 5), activation ='relu', padding = 'Same')(x)
	x = MaxPool2D(pool_size = (5, 5))(x)
	x = Dropout(0.2)(x)
	x = Flatten()(x)
	x = Dense(192, activation = "relu")(x)
	final_first_input_layer = Dropout(0.2)(x)

	second_input = Input(shape=(global_input_layer, ))

	# Merging features from images with pre-extracted features
	merge_one = concatenate([final_first_input_layer, second_input])

	x = Dense(global_hidden_layers[0], activation = 'relu')(merge_one)
	x = Dropout(0.2)(x)
	
	# For loop allows for easier adding and removing of hidden layers adjusting their sizes
	if (len(global_hidden_layers) > 1):
		for i in range(len(global_hidden_layers) - 1):
			x = Dense(global_hidden_layers[i + 1], activation = 'relu')(x)
			x = Dropout(0.2)(x)
	
	final = Dense(global_output_layer, activation = 'softmax')(x)
	model = Model(inputs=([first_input, second_input]), outputs=final)

	return model

def resize_img(img, max_dim):
	'''
	Resize the image so the maximum side is of size max_dim
	
	Parameters:
	img - image being resized
	max_dim - maximum dimension of image that was requested

	Returns a new image of the right size
	'''

    	# Get the axis with the larger dimension
    	max_ax = max((0, 1), key=lambda i: img.shape[i])

    	# Scale both axes so the image's largest dimension is max_dim
   	scale = max_dim / float(img.shape[max_ax])

    	return np.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)))

def reshape_img(images, max_dim, center = True):
	'''
	Reshapes images to the requested dimensions

	Parameters:
	images - list of the images that are being resized
	max_dim - maximum dimension of images that was requested
	center - whether or not the picture will be placed in the center when modified

	Returns a list of the modified images
	'''

	# Initialize the array that will hold the modified images
	modified = np.zeros((len(images), max_dim, max_dim, 1))

	# Reshape each image to the requested dimensions
	for i in range(len(images)):
		temp = resize_img(images[i], max_dim = max_dim)
		x = imresize(images[i], (temp.shape[0], temp.shape[1]), interp = 'nearest').reshape(temp.shape[0], 
			temp.shape[1], 1)

		length = x.shape[0]
		width = x.shape[1]

		# Place in center of True
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

def engineer_features():
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

	return

def apply_PCA():

	pca = PCA(0.80)
	pca.fit(train_mod_list.reshape(len(train_mod_list), 50 * 50))
	global_input_layer += pca.n_components_
	print pca.n_components_
	train_PCA = pca.transform(train_mod_list.reshape(len(train_mod_list), 50 * 50))
	test_PCA = pca.transform(test_mod_list.reshape(len(test_mod_list), 50 * 50))

	train = np.concatenate((train, train_PCA), axis = 1)
	test = np.concatenate((test, test_PCA), axis = 1)

	train = np.concatenate((train, train_width, train_height, train_asp_ratio, train_square), axis = 1)
	test = np.concatenate((test, test_width, test_height, test_asp_ratio, test_square), axis = 1)
	global_input_layer += 4
	return

def combined_generator(imgen, X):
    '''
    A generator to train our keras neural network. It
    takes the image augmenter generator and the array
    of the pre-extracted features.
    It yields a minibatch and will run indefinitely
    '''
    while True:
        for i in range(X.shape[0]):
            # Get the image batch and labels
            batch_img, batch_y = next(imgen)
            # This is where that change to the source code we
            # made will come in handy. We can now access the indicies
            # of the images that imgen gave us.
            x = X[imgen.index_array]
            yield [batch_img, x], batch_y

def augment_fit(mod, f1, t_list, x, y):
	'''
	Fitting our model that uses both augmented images and pre-extracted features 
	(Only applicable for the combined model)

	Parameters:
	mod - model used
	f1 - file name the best model is saved under
	t_list - list of modified training images
	x - matrix of training pre-extracted features
	y - matrix of testing pre-extracted features

	Returns the stats of the model
	'''

	print 'Using the augmented data fit_generator.'

	# Since fit_generator does not support validation_split we must create our own validation data first (10% of data)
	t_val = t_list[global_num_train - 99: global_num_train]
	x_val = x[global_num_train - 99: global_num_train]
	y_val = y[global_num_train - 99: global_num_train]
	
	# We need to remove the validation data from our test set
	t_train = t_list[0: global_num_train - 99]
	x_train = x[0: global_num_train - 99]
	y_train = y[0 : global_num_train - 99]

	# Creating our augmented data
	# ImageDataGenerator generates batches of tensor image data with real-time data augmentation.
	datagen = ImageDataGenerator2(rotation_range=20,
    		zoom_range=0.2,
   		horizontal_flip=True,
   		vertical_flip=True,
   		fill_mode='nearest')

	# Flow takes data & label arrays, generates batches of augmented data.
	x_batch = datagen.flow(t_train, y_train, batch_size = args.global_batch_size)	

	early_stopper= EarlyStopping(monitor = 'val_loss', patience = 280, verbose = 1, mode = 'auto')
	model_checkpoint = ModelCheckpoint(f1, monitor = 'val_loss', verbose = 1, save_best_only = True)

	spe = int(len(t_train) / args.global_batch_size)
	# fit_generator fits the model on batches with real-time data augmentation
	history = mod.fit_generator(combined_generator(x_batch, x_train), steps_per_epoch = spe, epochs = args.global_max_epochs,
		verbose = 0, validation_data = ([t_val, x_val], y_val),
		callbacks = [early_stopper, model_checkpoint])
	
	return history

def nn_fit(mod, f1, x, y):
	'''
	Fitting our model that uses only the pre-extracted features (Only applicable for the neural network model)

	Parameters:
	mod - model used
	f1 - file name the best model is saved under
	x - matrix of training pre-extracted features
	y - matrix of testing pre-extracted features

	Returns the stats of the model
	'''

	print 'Using the neural network fit.'
	early_stopper= EarlyStopping(monitor = 'val_loss', patience = 280, verbose = 1, mode = 'auto')
	model_checkpoint = ModelCheckpoint(f1, monitor = 'val_loss', verbose = 1, save_best_only = True)

	history = mod.fit(x, y, epochs = args.global_max_epochs, batch_size = args.global_batch_size,
		verbose = 0, validation_split = 0.15, shuffle = True, callbacks = [early_stopper, model_checkpoint])	

	return history

def combined_fit(mod, f1, t_list, x, y):
	'''
	Fitting our model that uses both image and pre-extracted features (Only applicable for the combined model)

	Parameters:
	mod - model used
	f1 - file name the best model is saved under
	t_list - list of modified training images
	x - matrix of training pre-extracted features
	y - matrix of testing pre-extracted features

	Returns the stats of the model
	'''

	print 'Using the combined network fit.'
	early_stopper= EarlyStopping(monitor = 'val_loss', patience = 280, verbose = 1, mode = 'auto')
	model_checkpoint = ModelCheckpoint(model_file, monitor = 'val_loss', verbose = 1, save_best_only = True)

	history = mod.fit(([t_list, x]), y, epochs = args.global_max_epochs,
		batch_size = args.global_batch_size, verbose = 0, validation_split = 0.15, shuffle = True,
		callbacks = [early_stopper, model_checkpoint])	

	return history	

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
			
# Grab the classes (will be used to set up our submission file)
classes = le.classes_
# Setting up one-hot scheme
y_train = np_utils.to_categorical(y)

# Extract the id of each leaf
train_ids = train.pop('id')

# Set up our testing data
test = pd.read_csv('data_provided/test.csv')

# Extract the id of each leaf
test_ids = test.pop('id')

# fit_transform() calculates the mean and std and also centers and scales data
x_train = StandardScaler().fit_transform(train)
x_test = StandardScaler().fit_transform(test)

# Load up all of the images
print 'Loading images...'
img_list, train_list, test_list = grab_images(train_ids, test_ids)
print 'Finished.'
	
# We need to reshape our images so they are all the same dimensions
train_mod_list = reshape_img(train_list, global_max_dim)
test_mod_list = reshape_img(test_list, global_max_dim)

# Look at images and some stats of leaves
if args.leaf_stats:
	print 'Displaying images...'
	visualize.image_similarity(train_mod_list, y, classes)
	visualize.visualize_leaves(train_mod_list, y, classes, show1 = True, show2 = True)
	print 'Finished.'

# Setting up the Keras neural network
# Choose a model
# model = create_model_softmax()
model = create_model_combined()

# Compile our model
sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
print model.summary()
#plot_model(model, to_file = 'modelCombined.png', show_shapes = True)

'''
Choose a fit for our model
Early stopping helps prevent over fitting by stopping our fitting function 
if our val_loss doesn't decrease after a certain number of epochs (called patience)
Model checkpoint saves the best weights obtained during training
'''
model_file = 'bestWeights6.hdf5'
# history = augment_fit(model, model_file, train_mod_list, x_train, y_train)
# history = nn_fit(model, model_file, x_train, y_train)
history = combined_fit(model, model_file, train_mod_list, x_train, y_train)

# Check Keras' statistics
if args.disp_stats:
	print 'Displaying stats...'
	visualize.plt_perf(history, filename1, filename2, p_loss = True, p_acc = True, val = True, save = args.save_stats)
	print 'Finished.'

# Reload our best weights
model.load_weights(model_file)

# Test our model on the test set
y_pred = model.predict([test_mod_list, x_test])
print '\n'

# FIXME ##################
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
##############################

# Set up the predictions into the correct format to submit to Kaggle
y_pred = pd.DataFrame(y_pred, index = test_ids, columns = classes)

# Save predictions to a csv file to submit
if args.save_prob:
	print 'Saving to file...'
	fp = open(filename3,'w')
	fp.write(y_pred.to_csv())

print 'Finished.'

