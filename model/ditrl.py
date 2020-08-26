import numpy as np 

# plan to always use the activation map and work back from there

class DITRL:
	def __init__(self, use_generated_files):
		self.output_file = None
		self.use_generated_files = use_generated_files

		self.threshold_values = None

		self.scaler = None
		self.TFIDF = None

	def forward(self, activation_map):
		# extract ITRs
		iad = self.convert_activation_map_to_IAD(activation_map)
		binarized_iad = self.binarize_IAD(iad)
		itr = self.convert_IAD_to_ITR(binarized_iad)

		# pre-process ITRS
		# scale / TFIDF

		# evaluate on ITR

	# ---
	# extract ITRs
	# ---

	def convert_activation_map_to_IAD(self, activation_map, save_name=""):
		# reshape activation map

		# perform max? 

		# update threshold

		# save as np file

		# return np

	def binarize_IAD(self, iad):
		# apply threshold

		# return iad

	def convert_IAD_to_ITR(self, binarized_iad):
		# apply threshold

		# return iad

	# ---
	# extract ITRs
	# ---