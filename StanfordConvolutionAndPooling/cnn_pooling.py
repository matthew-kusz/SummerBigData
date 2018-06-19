# Import the necessary packages
import numpy as np

def pooling(pool_dim, convolve_feat):
	'''
	Parameters:
	pool_dim - dimension of pooling region
	convolve_feat - convolved features to pool

	Returns:
	pooled_features - matrix of pooled features
	'''

	# Number of convolve_feats
	m = convolve_feat.shape[1]
	# Dimension of convolve_feat
	convolved_dim = convolve_feat.shape[2]
	# Number of channels
	num_features = convolve_feat.shape[0]
	print m, convolved_dim, num_features

	pooled_features = np.zeros((num_features, m, np.floor(convolved_dim / pool_dim), np.floor(convolved_dim / pool_dim))) # (400, m, 3, 3)
	print pooled_features.shape
