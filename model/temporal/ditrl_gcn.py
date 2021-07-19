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

from scipy import ndimage


from torch_geometric.data import Data

class DITRL_MaskFinder:
	def __init__(self):
		self.min_values = None
		self.max_values = None
		self.avg_values = None

		self.threshold_file_count = 0

	def add_data(self, iad):
		#print("iad:", iad.shape)
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

		#print("max_v:", self.max_values)
		#print("min_v:", self.min_values)

		return mask, threshold


class DITRL_Pipeline:
	def __init__(self, bottleneck_features, use_gcn=False):

		self.bottleneck_features = bottleneck_features

		self.is_training = False

		self.preprocessing = False
		self.data_store = []
		self.tfidf = TfidfTransformer(sublinear_tf=True)
		self.scaler = MinMaxScaler()
		self.trim_beginning_and_end = False
		self.smooth_with_savgol = False
		self.fs = True
		self.use_gcn = use_gcn

		self.mask_idx = np.arange(self.bottleneck_features)
		self.threshold_values = np.zeros(self.bottleneck_features)

	def convert_activation_map_to_itr(self, activation_map, cleanup=False):
		iad = self.convert_activation_map_to_iad(activation_map)
		print("iad.shape:", iad.shape)

		sparse_map = self.convert_iad_to_sparse_map(iad)
		print("sparse_map.shape:", len(sparse_map))

		if self.use_gcn:
			return self.convert_sparse_map_to_itr(sparse_map, iad=iad)
		itr = self.convert_sparse_map_to_itr(sparse_map, cleanup)
		itr = self.post_process(itr)

		#print("itr.shape:", itr.shape)

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
		#print("iad:", iad.shape)
		iad = iad[self.mask_idx]

		# trim start noisy start and end of IAD
		if self.trim_beginning_and_end:
			if iad.shape[1] > 10:
				iad = iad[:, 3:-3]

		# use savgol filter to smooth the IAD
		if self.smooth_with_savgol:
			for i in range(iad.shape[0]):
				iad[i] = savgol_filter(iad[i], 7, 1)

		return iad

	def convert_iad_to_sparse_map(self, iad):
		"""Convert the IAD to a sparse map that denotes the start and stop times of each feature"""


		#iad = iad[:, :400]
		# create mask and remove singletons and merge close segments
		mask = np.zeros_like(iad)
		max_values = self.threshold_values.reshape(len(self.mask_idx), 1)
		print(self.threshold_values)
		for i, row in enumerate(iad):
			#print(f"max_values[{i}]:", max_values[i])
			mask[i] = row > max_values[i]
			mask[i] = ndimage.binary_closing(mask[i])
			mask[i] = ndimage.binary_opening(mask[i])
			#print(mask[i])

		cnt = 0
		for r in range(mask.shape[0]):
			for c in range(1, mask.shape[1]):
				if mask[r, c] != mask[r, c - 1] and mask[r, c - 1] == 1:
					cnt += 1
			if mask[r, mask.shape[1] - 1] == 1:
				cnt += 1
		print("maskcnt:", cnt)

		#np.save("mask_new.npz", mask)
		#print("mask:", mask)
		# apply threshold to get indexes where features are active
		locs = np.where(mask)
		#locs = np.where(iad > self.threshold_values.reshape(len(self.mask_idx), 1))
		locs = np.dstack((locs[0], locs[1]))
		locs = locs[0]

		#old_mask = iad > self.threshold_values.reshape(len(self.mask_idx), 1)
		#np.save("mask_old.npz", old_mask)
		#assert False
		'''

		#print("new locs:", locs, locs.shape)

		locs = np.where(iad > self.threshold_values.reshape(len(self.mask_idx), 1))
		locs = np.dstack((locs[0], locs[1]))
		locs = locs[0]

		#print("old locs:", locs, locs.shape)

		assert False
		'''
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


	def find_relations(self, e1_t, e2_t):
		a1 = e1_t[0]
		a2 = e1_t[1]
		b1 = e2_t[0]
		b2 = e2_t[1]

		# before
		if (a2 < b1):
			return 0 # 'b';

		# meets
		if (a2 == b1):
			return 1 # 'm';

		# overlaps
		if (a1 < b1 and a2 < b2 and b1 < a2):
			return 2 # 'o';

		# during
		if (a1 < b1 and b2 < a2):
			return 3 # 'd';

		# finishes
		if (b1 < a1 and a2 == b2):
			return 4 # 'f';

		# starts
		if (a1 == b1 and a2 < b2):
			return 5 # 's';

		# equals
		if (a1 == b1 and a2 == b2):
			return 6 # 'e';
		return -1


	def convert_sparse_map_to_itr(self, sparse_map, iad=None):

		relations = []
		events = []

		for f1 in range(len(sparse_map)):
			for e1 in range(len(sparse_map[f1])):
				e1_l = str(f1)+"_"+str(e1)
				e1_t = sparse_map[f1][e1]
				#print("e1_t[0],e1_t[1]:", e1_t[0],e1_t[1])
				#print("iad[e1_t[0]:e1_t[1]]", iad[f1, e1_t[0]:e1_t[1]], iad.shape)
				#print("iad[e1_t[0]:e1_t[1]].max()", iad[f1, e1_t[0]:e1_t[1]].max())
				e1_weight = 1 if iad is None else iad[f1, e1_t[0]:e1_t[1]].max()
				#print("e1_weight:", e1_weight)
				events.append((e1_l, e1_weight))

				for f2 in range(len(sparse_map)):
					for e2 in range(len(sparse_map[f2])):
						e2_l = str(f2) + "_" + str(e2)
						e2_t = sparse_map[f2][e2]

						itr = self.find_relations(e1_t, e2_t)
						#if itr >= 0:
						if itr > 0 or (itr == 0 and f1 == f2):
							relations.append((e1_l, e2_l, itr))
		#return relations
		'''
		x = x.detach().cpu().numpy()[0]
		# print("x:", x.shape)
		edge_idx = []
		# edge_idx = set()  # [2, num_edges] edge connections (COO format)
		edge_attr = []  # [1, num_edges] type of relationship (ITR)
		node_x = np.zeros((self.node_size, self.node_size))
		# node_x = np.arange(self.node_size).reshape(-1, 1)

		for i in range(self.node_size):
			node_x[i, i] = 1
			for j in range(self.node_size):
				for itr in range(self.num_relations):
					if (x[i, j, itr] != 0):
						edge_idx.append((i, j))
						# edge_idx.add((i, j))
						edge_attr.append(itr)

		# edge_idx = np.array(list(edge_idx)).T
		edge_idx = np.array(edge_idx).T
		edge_attr = np.array(edge_attr)  # .reshape(1, -1)

		node_x = torch.autograd.Variable(torch.from_numpy(node_x).cuda()).float()
		edge_idx = torch.autograd.Variable(torch.from_numpy(edge_idx).cuda())
		edge_attr = torch.autograd.Variable(torch.from_numpy(edge_attr).cuda())
		'''
		e_map = {}

		node_x = np.zeros((len(events), len(sparse_map)))
		for e in range(len(events)):
			e_name = events[e][0]
			e_weight = events[e][1]
			#print("e_name:", e_name)
			#print("e_weight:", e_weight)

			e_map[e_name] = e
			node_x[e][int(e_name.split('_')[0])] = e_weight

		edge_idx = []
		edge_attr = []
		for r in relations:
			e1, e2, itr = r
			edge_idx.append((e_map[e1], e_map[e2]))
			edge_attr.append(itr)

		edge_idx = np.array(edge_idx).T
		edge_attr = np.array(edge_attr)

		#print("node_x:", node_x)
		#print("edge_idx:", edge_idx)
		#print("edge_attr:", edge_attr)

		return node_x, edge_idx, edge_attr#Data(node_x, edge_index=edge_idx, edge_attr=edge_atrr)



	def post_process(self, itr):
		# scale values to be between 0 and 1
		itr = itr.reshape(1, -1)

		#print("self.is_training:", self.is_training)
		if self.is_training:
			self.data_store.append(itr)
		else:
			itr = self.scaler.transform(itr)
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

