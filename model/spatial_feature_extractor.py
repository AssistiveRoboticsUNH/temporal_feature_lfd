import torch
import torch.nn as nn

from .backbone_model.tsm.ops.basic_ops import ConsensusModule

class SpatialFeatureExtractor(nn.Module):   
	def __init__(self, 
			num_classes, 
			use_aud, 
			is_training, 
			checkpoint_file, 
			num_segments,
		):

		super().__init__()

		self.num_classes = num_classes
		self.use_aud = use_aud
		self.is_training = is_training
		self.checkpoint_file = checkpoint_file
		self.num_segments = num_segments

		self.bottleneck_size = 128

		# rgb net
		from .backbone_model.tsm.tsm import TSMWrapper as VisualFeatureExtractor
		self.rgb_net = VisualFeatureExtractor(
			self.checkpoint_file, 
			self.num_classes, 
			training=is_training)

		self.linear_dimension = self.bottleneck_size
		'''
		# audio net
		if (self.use_aud):
			from aud.abc import AudioNetwork as AudioNetwork
			self.aud_net = AudioNetwork().getLogits()

			self.linear_dimension += self.aud_net.size()
		'''
		# pass to LSTM
		self.linear = nn.Sequential(
			nn.Linear(self.linear_dimension, self.num_classes)
		)

		self.consensus = ConsensusModule('avg')

	# Defining the forward pass    
	def forward(self, rgb_x):

		# pass data through CNNs
		rgb_y = self.rgb_net(rgb_x)
		#rgb_y = self.maxpool(rgb_y)
		print("rgb_y size:", rgb_y.size())

		# if using audio data as well I need to combine those features
		'''
		if (self.use_aud):
			aud_y = self.aud_net(aud_x)
			obs_x = torch.stack([rgb_y, aud_y], dim=0, out=None)
		else:
			obs_x = rgb_y
		'''

		# pass through linear layer
		base_out = rgb_y
		
		if self.rgb_net.is_shift and self.rgb_net.temporal_pool:
			base_out = base_out.view((-1, self.rgb_net.num_segments // 2) + base_out.size()[1:])
		else:
			base_out = base_out.view((-1, self.rgb_net.num_segments) + base_out.size()[1:])
		output = self.consensus(base_out)

		obs_y = self.linear(output)
		obs_y = self.consensus(obs_y)
		print("obs_y size:", obs_y.size())

		return obs_y