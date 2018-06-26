# Import the necessary packages
import numpy as np

def pooling(pool_dim, convolve_feat):
	'''
	Parameters:
	pool_dim - dimension of pooling region (2)
	convolve_feat - convolved features to pool (100, m, 14, 14)

	Returns:
	pooled_features - matrix of pooled features (100, m, 7, 7)
	'''

	# Number of convolve_feats
	m = convolve_feat.shape[1]
	# Dimension of convolve_feat
	convolved_dim = convolve_feat.shape[2]
	# Number of features
	num_features = convolve_feat.shape[0]

	pooled_features = np.zeros((num_features, m, np.floor(convolved_dim / pool_dim), np.floor(convolved_dim / pool_dim))) # (100, m, 7, 7)

	# We need to divide our convolved features into 4 different sections and find the mean feature activation over these regions
	# For each row
	for j in range(convolved_dim / pool_dim):
		# For each column
		for k in range(convolved_dim / pool_dim):
			temp = np.zeros((num_features, m, pool_dim, pool_dim))
			temp = convolve_feat[:, :, j * pool_dim: pool_dim + j * pool_dim, k * pool_dim: pool_dim + k * pool_dim]
			pooled_features[:, : ,j, k] = np.mean(temp, axis = (2, 3))

	return pooled_features
