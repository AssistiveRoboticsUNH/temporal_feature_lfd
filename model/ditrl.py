import numpy as np 
from scipy.signal import savgol_filter

import torch
import torch.nn as nn

import tempfile
from .parser_utils import write_sparse_matrix



# plan to always use the activation map and work back from there

class DITRLWrapper(nn.Module):
	def __init__(self, num_features, num_classes, is_training):
		super().__init__()

		self.ditrl = DITRL(num_features, num_classes, is_training)

	def forward(self, activation_map):

		sparse_map_filename = tempfile.TemporaryFile()
		print("sparse_map_filename:", sparse_map_filename)

		activation_map = activation_map.detach().cpu().numpy()
		iad 		= self.ditrl.convert_activation_map_to_IAD(activation_map)
		sparse_map  = self.ditrl.convert_IAD_to_sparse_map(iad, sparse_map_filename)
		itr 		= self.ditrl.convert_sparse_map_to_ITR(sparse_map_filename)
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

		iad = np.reshape(activation_map, (-1, self.num_features))
		iad = iad.T

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
			print(iad.shape, np.mean( iad , axis=1).shape)
			self.threshold_values += np.mean( iad , axis=1)
			self.threshold_file_count += 1

			self.threshold_values /= self.threshold_file_count

		# return IAD
		# ---
		return iad

	def convert_IAD_to_sparse_map(self, iad, sparse_map_filename):
		'''Convert the IAD to a sparse map that denotes the start and stop times of each feature'''

		# apply threshold
		# ---

		# threshold, reverse the locations to account for the transpose

		print("B: iad:", iad.shape)
		locs = np.where(iad > self.threshold_values.reshape(self.num_features, 1))
		locs = np.dstack((locs[1], locs[0]))

		# get the start and stop times for each feature in the IAD
		if(len(locs) != 0):
			sparse_map = []
			for i in range(iad.shape[0]):
				feature_row = locs[np.where(locs[:,0] == i)][:,1]

				# locate the start and stop times for the row of features
				start_stop_times = []
				if(len(feature_row) != 0):
					start = feature_row[0]
					for i in range(1, len(feature_row)):

						if( feature_row[i-1]+1 < feature_row[i] ):
							start_stop_times.append([start, feature_row[i-1]+1])
							start = feature_row[i]

					start_stop_times.append([start, feature_row[len(feature_row)-1]+1])

				# add start and stop times to sparse_map
				sparse_map.append( start_stop_times )
		else:
			sparse_map = [[] for x in xrange(iad.shape[0])]

		# write start_stop_times to file.
		# ---
	
		write_sparse_matrix(sparse_map_filename, sparse_map)

		# return sparse_map
		# ---
		return sparse_map

	def convert_sparse_map_to_ITR(self, sparse_map_filename):
		# execute c++ code

		import subprocess
		subprocess.call(["itr_parser", sparse_map_filename])

		#open file