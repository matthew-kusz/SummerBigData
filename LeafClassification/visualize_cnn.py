# Most of code is from: https://www.kaggle.com/abhmul/keras-convnet-lb-0-0052-w-visualization/notebook
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data_setup

from sklearn.preprocessing import StandardScaler # Preprocessing 
from keras import backend as K
from keras.models import load_model
from keras.layers import MaxPooling2D
from scipy.misc import imresize

global_num_train = 990
global_num_test = 594
NUM_LEAVES = 3
global_max_dim = 50
model_fn = 'bestWeights3.hdf5'

####### Definitions #######
# Function by gcalmettes from http://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(figures, nrows = 1, ncols=1, titles=False):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(sorted(figures.keys(), key=lambda s: int(s[3:]))):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray(), interpolation = 'none')
        if titles:
            axeslist.ravel()[ind].set_title(title)

    for ind in range(nrows*ncols):
        axeslist.ravel()[ind].set_axis_off()

    if titles:
        plt.tight_layout()
    plt.show()


def get_dim(num):
    """
    Simple function to get the dimensions of a square-ish shape for plotting
    num images
    """

    s = sqrt(num)
    if round(s) < s:
        return (int(s), int(s)+1)
    else:
        return (int(s)+1, int(s)+1)

# Load the best model
model = load_model(model_fn)

# Set up the data given to us
train_list, test_list, train_ids, test_ids, train, test, y, y_train, classes = data_setup.data()
	
# We need to reshape our images so they are all the same dimensions
train_mod_list = data_setup.reshape_img(train_list, global_max_dim)
test_mod_list = data_setup.reshape_img(test_list, global_max_dim)

# Grab more features to train on
train, test = data_setup.engineered_features(train, test, train_list, test_list)

train, test = data_setup.apply_PCA(train, test, train_mod_list, test_mod_list, global_max_dim, y)

# fit calculates the mean and transform centers and scales the data so we have 0 mean and unit variance
scaler = StandardScaler().fit(train)
x_train = scaler.transform(train)
x_test = scaler.transform(test)

# Set up our input layer size
global_input_layer = x_train.shape[1]

X_img_val = train_mod_list[global_num_train - 99: global_num_train]
X_num_val = x_train[global_num_train - 99: global_num_train]
y_val = y[global_num_train - 99: global_num_train]

# Get the convolutional layers
conv_layers = [layer for layer in model.layers if isinstance(layer, MaxPooling2D)]

# Pick random images to visualize
imgs_to_visualize = np.random.choice(np.arange(0, len(X_img_val)), NUM_LEAVES)

# Use a keras function to extract the conv layer data
convout_func = K.function([model.layers[0].input, K.learning_phase()], [layer.output for layer in conv_layers])
conv_imgs_filts = convout_func([X_img_val[imgs_to_visualize], 0])
# Also get the prediction so we know what we predicted
predictions = model.predict([X_img_val[imgs_to_visualize], X_num_val[imgs_to_visualize]])

imshow = plt.imshow #alias
# Loop through each image disply relevant info
for img_count, img_to_visualize in enumerate(imgs_to_visualize):

    # Get top 3 predictions
    top3_ind = predictions[img_count].argsort()[-3:]
    top3_species = np.array(classes)[top3_ind]
    top3_preds = predictions[img_count][top3_ind]

    # Get the actual leaf species
    actual = classes[y_val[img_to_visualize]]

    # Display the top 3 predictions and the actual species
    print("Top 3 Predicitons:")
    for i in range(2, -1, -1):
        print("\t%s: %s" % (top3_species[i], top3_preds[i]))
    print("\nActual: %s" % actual)

    # Show the original image
    plt.title("Image used: #%d (digit=%d)" % (img_to_visualize, y_val[img_to_visualize]))
    imshow(X_img_val[img_to_visualize][:, :, 0], cmap='gray', interpolation = 'none')
    plt.tight_layout()
    plt.show()

    # Plot the filter images
    for i, conv_imgs_filt in enumerate(conv_imgs_filts):
        conv_img_filt = conv_imgs_filt[img_count]
        print("Visualizing Convolutions Layer %d" % i)
        # Get it ready for the plot_figures function
        fig_dict = {'flt{0}'.format(i): conv_img_filt[:, :, i] for i in range(conv_img_filt.shape[-1])}
        plot_figures(fig_dict, *get_dim(len(fig_dict)))
