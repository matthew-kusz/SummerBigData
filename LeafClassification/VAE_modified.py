'''Example of VAE on leaves dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate leaves by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''
from keras.layers import Lambda, Input, Dense, Conv2DTranspose, Conv2D, Dropout, MaxPool2D, Flatten, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from sklearn.preprocessing import StandardScaler # Preprocessing

import data_setup
import pandas as pd  
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import argparse

####### Global variables #######
parser = argparse.ArgumentParser()
parser.add_argument('load_model', 
	help = 'Use 1/0 (True/False) to indicate if you want to load a previous model or not.', type = int)
parser.add_argument('disp_stats', 
	help = 'Use 1/0 (True/False) to indicate if you want to display model stats or not.', type = int)
parser.add_argument('global_max_epochs', help = 'Max amount of epochs allowed.', type = int)
parser.add_argument('global_batch_size', help = 'Number of samples per gradient update.', type = int)

args = parser.parse_args()

global_num_train = 990
global_num_test = 594
global_max_dim = 50

####### Definitions #######
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_leaves"):
    	"""Plots labels and leaves as function of 2-dim latent vector
    	# Arguments:
        	models (tuple): encoder and decoder models
        	data (tuple): test data and label
        	batch_size (int): prediction batch size
       		model_name (string): which model is using this function
 	"""

    	encoder, decoder = models
    	x_test, y_test, te_test = data
	print x_test.shape
	print y_test.shape
	
    	# Give a list of the id associated with each species
    	for i in range(len(classes)):
		print i, classes[i]
		
    	# Ask for what species to display	 
    	species = input('What leaf species would you like to look at? ')
   	print 'You chose', classes[species]

    	temp = np.where(y_test == species)
    	temp1 = np.zeros((len(temp[0]), global_max_dim * global_max_dim))
	temp2 = np.ones(len(temp[0])) * 99
    	print temp1.shape
    	print x_test.shape
    	for j in range(len(temp[0])):
	 	temp1[j] = x_test[temp[0][j],:]

	# temp2 = np.ones(len(te_test)) * 99
	x_test = np.vstack((x_test, temp1))
	y_test = np.concatenate((y_test, temp2), axis = 0)

    	filename = 'VAE/vae_leaves/vae_mean.png'

	custom_cmap = matplotlib.cm.get_cmap('brg')
	custom_cmap.set_over('yellow')

    	# display a 2D plot of the leaf classes in the latent space
    	z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)

    	plt.figure(figsize=(12, 10))
    	black_marker = mpatches.Circle(4, radius = 100, color = 'yellow', label = classes[species])
	plt.legend(handles=[black_marker], loc = 'best')
    	plt.scatter(z_mean[:, 0], z_mean[:, 1], c = y_test, cmap=custom_cmap)
	plt.clim(0,98)
    	plt.colorbar()
   	plt.xlabel("z[0]")
    	plt.ylabel("z[1]")
    	plt.savefig(filename)
    	# plt.show()

	filename = 'VAE/vae_leaves/vae_mean_zoomed.png'
	plt.figure(figsize=(12, 10))
   	black_marker = mpatches.Circle(8, radius = 100, color = 'yellow', label = classes[species])
	plt.legend(handles=[black_marker], loc = 'best')
    	plt.scatter(z_mean[:, 0], z_mean[:, 1], c = y_test, cmap=custom_cmap)
	plt.clim(0,98)
    	plt.colorbar()
   	plt.xlabel("z[0]")
    	plt.ylabel("z[1]")
	plt.xlim(-1, 1)
	plt.ylim(-0.75, 0.5)
    	plt.savefig(filename)
    	# plt.show()

    	filename = 'VAE/vae_leaves/leaves_over_latent.png'
    	# display a 30x30 2D manifold of leaves
    	n = 30
    	leaf_size = global_max_dim
    	figure = np.zeros((leaf_size * n, leaf_size * n))
    	# linearly spaced coordinates corresponding to the 2D plot
    	# of leaf classes in the latent space
    	grid_x = np.linspace(-1, 1, n)
    	grid_y = np.linspace(-0.75, 0.5, n)[::-1]

    	for i, yi in enumerate(grid_y):
        	for j, xi in enumerate(grid_x):
            		z_sample = np.array([[xi, yi]])
            		x_decoded = decoder.predict(z_sample)
            		leaf = x_decoded[0].reshape(leaf_size, leaf_size)
            		figure[i * leaf_size: (i + 1) * leaf_size,
                   		j * leaf_size: (j + 1) * leaf_size] = leaf

    	plt.figure(figsize=(10, 10))
    	start_range = leaf_size // 2
    	end_range = n * leaf_size + start_range + 1
    	pixel_range = np.arange(start_range, end_range, leaf_size)
    	sample_range_x = np.round(grid_x, 1)
    	sample_range_y = np.round(grid_y, 1)
    	plt.xticks(pixel_range, sample_range_x)
    	plt.yticks(pixel_range, sample_range_y)
    	plt.xlabel("z[0]")
    	plt.ylabel("z[1]")
    	plt.imshow(figure, cmap='Greys_r')
    	plt.savefig(filename)
    	plt.show()

######## Code ########
# Set up the data given to us
train_list, test_list, train_ids, test_ids, train, test, y, y_train, classes = data_setup.data()

# fit_transform() calculates the mean and std and also centers and scales data
x_train = StandardScaler().fit_transform(train)
x_test = StandardScaler().fit_transform(test)

# We need to reshape our images so they are all the same dimensions
train_mod_list = data_setup.reshape_img(train_list, global_max_dim)
test_mod_list = data_setup.reshape_img(test_list, global_max_dim)

# Grab the dimensions we are using for our images
image_size = global_max_dim
# Find the flattened size
original_dim = image_size * image_size

# Set up our validation set (10% of data)
tr_val = train_mod_list[global_num_train - 99: global_num_train]
x_val = x_train[global_num_train - 99: global_num_train]
y_val = y_train[global_num_train - 99: global_num_train]
	
# We need to remove the validation data from our test set
tr_train = train_mod_list[0: global_num_train - 99]
x_train = x_train[0: global_num_train - 99]
y_train = y_train[0: global_num_train - 99]

# For graphing purposes
y = y[0: global_num_train - 99]

# Reshape our images to this size
tr_train = tr_train.reshape(-1, original_dim)
tr_val = tr_val.reshape(-1, original_dim)
te_test = test_mod_list.reshape(-1, original_dim)

# network parameters
input_shape = (original_dim, )
batch_size = args.global_batch_size
latent_dim = 2
epochs = args.global_max_epochs

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(512, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(320, activation = "relu")(x)
x = Dropout(0.2)(x)
x = Dense(240, activation = "relu")(x)
x = Dropout(0.2)(x)
x = Dense(192, activation = "relu")(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='VAE/vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(192, activation = "relu")(latent_inputs)
x = Dropout(0.2)(x)
x = Dense(240, activation = "relu")(x)
x = Dropout(0.2)(x)
x = Dense(320, activation = "relu")(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='VAE/vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

models = (encoder, decoder)
data = (tr_train, y, te_test)

# VAE loss = mse_loss or xent_loss + kl_loss
reconstruction_loss = mse(inputs, outputs)
# reconstruction_loss = binary_crossentropy(inputs, outputs)

reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam', loss = None)
vae.summary()
# plot_model(vae, to_file='VAE/vae_mlp.png', show_shapes=True)

model_file = 'VAE/vae_mlp_leaves_weights_nn_dim50-3.h5'
if args.load_model:
	# load weights from a previous run
	print 'Loading weights...'
	vae.load_weights(model_file)
else:
	# train the autoencoder
	print 'Training model...'
	
	model_checkpoint = ModelCheckpoint(model_file, monitor = 'val_loss', verbose = 1, save_best_only = True)
	
        vae.fit(tr_train, y_train,
		shuffle = True,
		verbose = 0,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(tr_val, y_val),
		callbacks = [model_checkpoint])

if args.disp_stats:
	plot_results(models, data, batch_size=batch_size, model_name="vae_mlp")

