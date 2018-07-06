# Import the necessary packages
import numpy as np
import matplotlib as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

####### Global Variables #######


####### Definitions #######


####### Code #######
# We need to extract the data given
data = np.genfromtxt('data_provided/train.csv.zip')
print data
