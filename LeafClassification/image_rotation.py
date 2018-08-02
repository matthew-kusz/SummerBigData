# Import the necessary packages
import numpy as np
from scipy.ndimage import rotate

from sklearn.metrics.pairwise import cosine_similarity   # Used for checking how similar images are to one another

# Set up the data given to us
train_list, test_list, train_ids, test_ids, train, test, y, y_train, classes = data_setup.data()

# We need to reshape our images so they are all the same dimensions
train_mod_list = data_setup.reshape_img(train_list, global_max_dim)
test_mod_list = data_setup.reshape_img(test_list, global_max_dim)

# Let's orient them so they are all facing the same way


