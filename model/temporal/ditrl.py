# write/read binary files for ITR extraction
from model.parser_utils import write_sparse_matrix, read_itr_file

from sklearn.linear_model import SGDClassifier
from multiprocessing import Pool

# pre-processing functions
from scipy.signal import savgol_filter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import torch
import torch.nn as nn

import os
import subprocess
import tempfile


class DITRL_MaskFinder:
	def __init__(self):
		self.min_values = None
		self.max_values = None
		self.avg_values = None

		self.threshold_file_count = 0

	def add_data(self, iad):
		max_v = np.max(iad, axis=0)
		min_v = np.min(iad, axis=0)
		avg_v = np.mean(iad, axis=0)

		if self.min_values is None:
			self.min_values = min_v
			self.max_values = max_v
			self.avg_values = avg_v
		else:
			for i in range(len(max_v)):
				if max_v[i] > self.max_values[i]:
					self.max_values[i] = max_v[i]
				if min_v[i] > self.min_values[i]:
					self.min_values[i] = min_v[i]

			self.avg_values *= self.threshold_file_count
			self.avg_values += avg_v

		self.threshold_file_count += 1
		self.avg_values /= self.threshold_file_count

	def gen_mask_and_threshold(self):
		mask = np.where(self.max_values != self.min_values)[0]
		threshold = self.avg_values[mask]

		print("max_v:", self.max_values)
		print("min_v:", self.min_values)

		return mask, threshold


class DITRL_Pipeline:
	def __init__(self, bottleneck_features):

		self.bottleneck_features = bottleneck_features

		self.is_training = False

		self.preprocessing = False
		self.data_store = []
		self.tfidf = TfidfTransformer(sublinear_tf=True)
		self.scaler = self.tfidf#MinMaxScaler()
		self.trim_beginning_and_end = False
		self.smooth_with_savgol = False
		self.fs = True

		self.mask_idx = np.arange(self.bottleneck_features)
		self.threshold_values = np.zeros(self.bottleneck_features)

	def convert_activation_map_to_itr(self, activation_map, cleanup=False):
		iad = self.convert_activation_map_to_iad(activation_map)
		print("iad.shape:", iad.shape)

		sparse_map = self.convert_iad_to_sparse_map(iad)
		print("sparse_map.shape:", len(sparse_map))

		itr = self.convert_sparse_map_to_itr(sparse_map, cleanup)
		itr = self.post_process(itr)

		print("itr.shape:", itr.shape)

		itr = itr.astype(np.float32)
		return itr

	def convert_activation_map_to_iad(self, activation_map):
		# reshape activation map
		# ---


		iad = np.reshape(activation_map, (-1, self.bottleneck_features))
		iad = iad.T

		# pre-processing of IAD
		# ---

		# mask unnecessary features
		iad = iad[self.mask_idx]

		# trim start noisy start and end of IAD
		if self.trim_beginning_and_end:
			if iad.shape[1] > 10:
				iad = iad[:, 3:-3]

		# use savgol filter to smooth the IAD
		if self.smooth_with_savgol:
			for i in range(iad.shape[0]):
				iad[i] = savgol_filter(iad[i], 3, 1)

		return iad

	def convert_iad_to_sparse_map(self, iad):
		"""Convert the IAD to a sparse map that denotes the start and stop times of each feature"""

		# apply threshold to get indexes where features are active
		locs = np.where(iad > self.threshold_values.reshape(len(self.mask_idx), 1))
		locs = np.dstack((locs[0], locs[1]))
		locs = locs[0]

		# get the start and stop times for each feature in the IAD
		if len(locs) != 0:
			sparse_map = []
			for i in range(iad.shape[0]):
				feature_row = locs[np.where(locs[:, 0] == i)][:, 1]

				# locate the start and stop times for the row of features
				start_stop_times = []
				if len(feature_row) != 0:
					start = feature_row[0]
					for j in range(1, len(feature_row)):
						if feature_row[j-1]+1 < feature_row[j]:
							start_stop_times.append([start, feature_row[j-1]+1])
							start = feature_row[j]

					start_stop_times.append([start, feature_row[len(feature_row)-1]+1])

				# add start and stop times to sparse_map
				sparse_map.append(start_stop_times)
		else:
			sparse_map = [[] for x in range(iad.shape[0])]

		return sparse_map

	def convert_sparse_map_to_itr(self, sparse_map, cleanup=True):

		# create files
		file_id = next(tempfile._get_candidate_names())
		sparse_map_filename = os.path.join("/tmp", file_id+".b1")
		itr_filename = os.path.join("/tmp", file_id+".b2")

		# write the sparse map to a file
		write_sparse_matrix(sparse_map_filename, sparse_map)

		# execute the itr identifier (C++ code)
		try:
			subprocess.call(["model/itr_parser", sparse_map_filename, itr_filename])
		except():
			print("ERROR: ditrl.py: Unable to extract ITRs from sparse map, did you generate the C++ executable?")
			# if not go to 'models' directory and type 'make'

		#open ITR file
		itrs = read_itr_file(itr_filename)

		#file cleanup
		if cleanup:
			os.system("rm "+sparse_map_filename+" "+itr_filename)

		return itrs

	def post_process(self, itr):
		# scale values to be between 0 and 1
		itr = itr.reshape(1, -1)

		#print("self.is_training:", self.is_training)
		if self.is_training:
			self.data_store.append(itr)
		else:
			itr = self.scaler.transform(itr)

		print("type(itr):", type(itr))
		return itr

	def fit_tfidf(self):
		if self.data_store is not None:
			print("len(data_store):", len(self.data_store), self.data_store[0])
			self.data_store = np.array(self.data_store).squeeze(1)
			print("self.data_store.shape:", self.data_store.shape)
			self.scaler.fit(self.data_store)
			self.data_store = None


class DITRL_Linear(nn.Module):
	def __init__(self, num_features, num_classes, is_training, model_name):
		super().__init__()

		self.inp_dim = num_features * num_features * 7
		self.num_classes = num_classes
		self.model_name = model_name

		self.model = nn.Sequential(
			nn.Linear(self.inp_dim, self.num_classes)
		)

		# load a previously saved model
		if not is_training:
			ext_checkpoint = self.model_name
			if ext_checkpoint:

				# load saved model parameters
				print("ditrl.py: Loading Extension Model from: ", ext_checkpoint)
				checkpoint = torch.load(ext_checkpoint)

				self.model.load_state_dict(checkpoint, strict=True)

				# prevent changes to these parameters
				for param in self.model.parameters():
					param.requires_grad = False
			else:
				print("ditrl.py: Did Not Load Extension Model")

	def forward(self, data):
		#print("data input_shape:", data.shape)
		data = torch.reshape(data, (-1, self.inp_dim))
		return self.model(data)

	def save_model(self):
		torch.save(self.model.state_dict(), self.model_name)
		print("Ext model saved to: ", self.model_name)

'''
class DITRL_SVM:
	def __init__(self, num_features, num_classes, is_training):
		alpha = 0.001
		n_jobs = 4
		self.model = SGDClassifier(loss='hinge', alpha=alpha, n_jobs=n_jobs)

		self.num_classes = num_classes

	def forward(self, data):
		data = scipy.sparse.coo_matrix(data)
		return self.model.predict(data)

	def train(self, data, label):
		data = scipy.sparse.coo_matrix(data)
		label = np.array(label)
		net.partial_fit(data, label, classes=np.arange(self.num_classes))
'''

