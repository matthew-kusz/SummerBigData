# Import the necessary packages
# Misc
import numpy as np				 	 # Allow for easier use of arrays and linear algebra
import matplotlib                                	 # Imported for use of matplotlib.use('Agg')
matplotlib.use('Agg')                            	 # Use if submitting job to the supercomputer
import pandas as pd                         	 	 # For reading in and writing files
import argparse                                  	 # For inputing values outside of the code
import visualize                                 	 # Python code for visualizing images
import data_setup                                        # Python code for setting up the data

# sklearn
from sklearn.preprocessing import StandardScaler 	 # Preprocessing
from sklearn.model_selection import StratifiedKFold      # K-fold validation

# Keras
from keras.models import Sequential, Model      	 # Linearly sets up model, allows the use of keras' API
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, concatenate, Input
from keras.utils import plot_model            		 # Visualizing model
from keras.callbacks import EarlyStopping 	         # Used to prevent overfitting
from keras.callbacks import ModelCheckpoint              # Gives us the best weights obtained during fitting
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
parser.add_argument('global_batch_size', help = 'Number of samples per gradient update.', type = int)

args = parser.parse_args()

global_num_train = 990
global_num_test = 594
global_hidden_layers = [200, 100] #[400, 200]
global_output_layer = 99
global_num_classes = 99
global_max_dim = 50

filename1 = 'graphs/lossEpoch' + str(args.global_max_epochs) + 'Batch' + str(args.global_batch_size) + 'Softmax.png'
filename2 = 'graphs/accEpoch' + str(args.global_max_epochs) + 'Batch' + str(args.global_batch_size) + 'Softmax.png'
filename3 = 'submissions/submissionEpoch' + str(args.global_max_epochs) + 'Batch' + str(args.global_batch_size) + 'Softmax.csv'
model_file = 'bestWeights2.hdf5'

# Set up a seed so that our results don't fluctuate over multiple tests
seed = 4
np.random.seed(seed)

####### Definitions #######
def create_model_softmax():
	'''
	This model uses only the pre-extracted features of the leaves to create a softmax regression model
	Returns the built model
	'''
		
	mod1 = Sequential()	
	
	mod1.add(Dense(global_hidden_layers[0], input_dim = input_layer, activation = 'relu'))
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
	This model uses both the images and the pre-extracted features of the leaves to create a CNN model
	Returns the built model
	'''
	# Obtaining features from the images
	first_input = Input(shape=(global_max_dim, global_max_dim, 1))
	x = Conv2D(filters = 16, kernel_size = (5, 5), activation ='relu', padding = 'Same')(first_input)
	x = MaxPool2D(pool_size = (2, 2))(x)
	x = Dropout(0.2)(x)
	x = Conv2D(filters = 32, kernel_size = (5, 5), activation ='relu', padding = 'Same')(x)
	x = MaxPool2D(pool_size = (2, 2))(x)
	x = Dropout(0.2)(x)
	x = Flatten()(x)
	x = Dense(192, activation = 'relu')(x)
	final_first_input_layer = Dropout(0.2)(x)	

	# Merging features from images with pre-extracted features
	second_input = Input(shape=(input_layer, ))
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

	# Creating our augmented data
	# ImageDataGenerator generates batches of tensor image data with real-time data augmentation.
	datagen = ImageDataGenerator(rotation_range=20,
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
	history = mod.fit_generator(combined_generator(x_batch, x_train), steps_per_epoch = spe,
		epochs = args.global_max_epochs, verbose = 0, validation_data = ([t_val, x_val], y_val),
		callbacks = [early_stopper, model_checkpoint])
	
	return history

def nn_fit(mod, f1, x, y, tr, val, m_s, y_p, x_t):
	'''
	Fitting our model that uses only the pre-extracted features (Only applicable for the neural network model)

	Parameters:
	mod - model used
	f1 - file name the best model is saved under
	x - matrix of training pre-extracted features
	y - matrix of testing pre-extracted features
	tr- indices for the data that will be used for the training set
	val - indices for the data that will be used from the validation set
	m_s - array of scores from each run
	y_p - array of predictions from each run
	x_t - matrix of testing set with pre-extracted features + any additional features

	Returns:
	history - stats of the model
	m_s - array of scores from each run
	y_p - array of predictions from each run
	'''

	'''
	Early stopping helps prevent over fitting by stopping our fitting function 
	if our val_loss doesn't decrease after a certain number of epochs (called patience)
	Model checkpoint saves the best weights obtained during training
	'''
	print 'Using the neural network fit.'
	early_stopper= EarlyStopping(monitor = 'val_loss', patience =300, verbose = 1, mode = 'auto')
	model_checkpoint = ModelCheckpoint(f1, monitor = 'val_loss', verbose = 0, save_best_only = True)

	history = mod.fit(x[tr], y[tr], epochs = args.global_max_epochs, batch_size = args.global_batch_size,
		verbose = 0, validation_data = (x[val], y[val]), callbacks = [early_stopper, model_checkpoint])	

	# Grab the best weights from current run
	model.load_weights(f1)

	# Find the scores
	scores = mod.evaluate(x[val], y[val], verbose = 0)
	print '%s: %.5g' % (model.metrics_names[0], scores[0])

	# Store the scores for later
	m_s.append(scores[0])

	# Store the predictions for later
	y_p.append(model.predict(x_t))
	
	return history, m_s, y_p

def combined_fit(mod, f1, tr_list, te_list, x, y, tr, val, m_s, y_p, x_t):
	'''
	Fitting our model that uses both image and pre-extracted features (Only applicable for the combined model)

	Parameters:
	mod - model used
	f1 - file name the best model is saved under
	t_list - list of modified training images
	x - matrix of training pre-extracted features
	y - matrix of testing pre-extracted features
	tr- indices for the data that will be used for the training set
	val - indices for the data that will be used from the validation set
	m_s - array of scores from each run
	y_p - array of predictions from each run
	x_t - matrix of testing set with pre-extracted features + any additional features

	Returns:
	history - stats of the model
	m_s - array of scores from each run
	y_p - array of predictions from each run
	'''

	'''
	Early stopping helps prevent over fitting by stopping our fitting function 
	if our val_loss doesn't decrease after a certain number of epochs (called patience)
	Model checkpoint saves the best weights obtained during training
	'''
	print 'Using the combined network fit.'
	early_stopper= EarlyStopping(monitor = 'val_loss', patience = 300, verbose = 1, mode = 'auto')
	model_checkpoint = ModelCheckpoint(model_file, monitor = 'val_loss', verbose = 1, save_best_only = True)

	history = mod.fit(([tr_list[tr], x[tr]]), y[tr], epochs = args.global_max_epochs,
		batch_size = args.global_batch_size, verbose = 0,
		validation_data = ([tr_list[val], x[val]], y[val]),
		callbacks = [early_stopper, model_checkpoint])	

	# Grab the best weights from current run
	model.load_weights(f1)

	# Find the scores
	scores = mod.evaluate([tr_list[val], x[val]], y[val], verbose = 0)
	print '%s: %.5g' % (model.metrics_names[0], scores[0])

	# Store the scores for later
	m_s.append(scores[0])

	# Store the predictions for later
	y_p.append(model.predict([te_list, x_t]))
	
	return history, m_s, y_p

####### Code #######
# Set up the data given to us
train_list, test_list, train_ids, test_ids, train, test, y, y_train, classes = data_setup.data()

# Grab more features to train on
train, test = data_setup.engineered_features(train, test, train_list, test_list)

# Grab even more features with openCV
train, test = data_setup.more_features(train, test, train_list, test_list)

# We need to reshape our images so they are all the same dimensions
train_mod_list = data_setup.reshape_img(train_list, global_max_dim)
test_mod_list = data_setup.reshape_img(test_list, global_max_dim)

# Let's apply PCA to the images and attach them to the pre-extracted features
train, test = data_setup.apply_PCA(train, test, train_mod_list, test_mod_list, global_max_dim, y, classes)

# fit calculates the mean and transform centers and scales the data so we have 0 mean and unit variance
scaler = StandardScaler().fit(train)
x_train = scaler.transform(train)
x_test = scaler.transform(test)

# Look at images and some stats of leaves
if args.leaf_stats:
	print 'Displaying images...'
	visualize.image_similarity(train_mod_list, y, classes)
	visualize.visualize_leaves(train_mod_list, y, classes, show1 = False, show2 = True)
	print 'Finished.'

# Set up our input layer size
input_layer = x_train.shape[1]

# We will use 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# Set up arrays to store the score and predictions for each iteration
mean_score = []
y_pred = []

# Setting up the Keras neural network
for training, validation in kfold.split(x_train, y):
	print 'Training...'
	# Choose a model
	model = create_model_softmax()
	# model = create_model_combined()

	# Compile our model
	sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
	model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

	#Choose a fit for our model
	'''
	history, mean_score, y_pred = augment_fit(model, model_file, train_mod_list, x_train, y_train, training
						validation, mean_score, y_pred, x_test)
	'''
	
	history, mean_score, y_pred = nn_fit(model, model_file, x_train, y_train, training, validation,
						mean_score, y_pred, x_test)
	
	'''
	history = combined_fit(model, model_file, train_mod_list, test_mod_list, x_train, y_train, training,
						validation, mean_score, y_pred, x_test)
	'''

# Print the average validation lost and its standard deviation
print '%.5g (+/- %.3g)' % (np.mean(mean_score), np.std(mean_score))
print model.summary()
# plot_model(model, to_file = 'modelSoftmax2.png', show_shapes = True)

# Check Keras' statistics
if args.disp_stats:
	print 'Displaying stats...'
	visualize.plt_perf(history, filename1, filename2, p_loss = True, p_acc = True, val = True,
		save = args.save_stats)
	print 'Finished.'

# Average our predictions over all of the runs
all_pred = np.zeros((y_pred[0].shape[0], y_pred[0].shape[1]))
for i in range(len(y_pred)):
	all_pred += y_pred[i]

all_pred /= 10

# Save our numpy array to a .npy file for later use
np.save('NN_all_pred', all_pred)

# Set up the predictions into the correct format to submit to Kaggle
all_pred = pd.DataFrame(all_pred, index = test_ids, columns = classes)

# Save predictions to a csv file to submit
if args.save_prob:
	print 'Saving to file', filename3, '...'
	fp = open(filename3,'w')
	fp.write(all_pred.to_csv())

print 'Finished.'

