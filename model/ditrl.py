import numpy as np 
from scipy.signal import savgol_filter

import torch
import torch.nn as nn

# plan to always use the activation map and work back from there

class DITRLWrapper(nn.Module):
	def __init__(self, num_features, num_classes, is_training):
		super().__init__()

		self.ditrl = DITRL(num_features, num_classes, is_training)

	def forward(self, activation_map):
		activation_map = activation_map.detach().cpu().numpy()
		iad 		= self.ditrl.convert_activation_map_to_IAD(activation_map)
		sparse_map  = self.ditrl.convert_IAD_to_sparse_map(iad)
		itr 		= self.ditrl.convert_sparse_map_to_ITR(sparse_map)
		return itr

		# pre-process ITRS
		# scale / TFIDF

		# evaluate on ITR

class DITRL: # pipeline
	def __init__(self, num_features, num_classes, is_training):
		self.output_file = None
		self.use_generated_files = None

		self.num_features = num_features
		self.threshold_values = np.zeros(self.num_features, np.float32)
		self.threshold_file_count = 0
		self.num_classes = num_classes

		self.scaler = None
		self.TFIDF = None

		self.is_training = is_training

	# ---
	# extract ITRs
	# ---

	def convert_activation_map_to_IAD(self, activation_map, save_name="", ):
		# reshape activation map
		# ---

		print("activation_map:", activation_map.shape)
		iad = np.reshape(activation_map, (-1, self.num_features))

		# pre-processing of IAD
		# ---

		# trim start noisy start and end of IAD
		if(iad.shape[1] > 10):
			iad = iad[:, 3:-3]

		# use savgol filter to smooth the IAD
		smooth_window = 35
		if(iad.shape[1] > smooth_window):
			for i in range(iad.shape[0]):
				iad[i] = savgol_filter(iad[i], smooth_window, 3)

		# update threshold
		# ---
		if (self.is_training):

			altered_value = self.threshold_values * self.threshold_file_count
			self.threshold_values += np.mean( iad , axis=1)
			print(iad.shape, np.mean( iad , axis=1).shape)
			self.threshold_file_count += 1

			self.threshold_values /= self.threshold_file_count

		# save as IAD as npy file
		# ---



		# return IAD
		# ---
		return iad

	def convert_IAD_to_sparse_map(self, iad):
		'''Convert the IAD to a sparse map that denotes the start and stop times of each feature'''

		# apply threshold
		# ---

		# threshold, reverse the locations to account for the transpose

		locs = np.where(iad.T > threshold_values)
		locs = np.array( zip( locs[1], locs[0] ) )

		# get the start and stop times for each feature in the IAD
		if(len(locs) != 0):
			sparse_map = []
			for i in range(iad.shape[0]):
				feature_times = locs[np.where(locs[:,0] == i)][:,1]
				sparse_map.append( find_start_stop( feature_times ))
		else:
			sparse_map = [[] for x in xrange(iad.shape[0])]

		# write start_stop_times to file.
		# ---
		
		#write_sparse_matrix(ex['b_path'], sparse_map)

		# return sparse_map
		# ---
		return sparse_map

	def convert_sparse_map_to_ITR(self, sparse_map):
		# execute c++ code
		pass

		#return itr