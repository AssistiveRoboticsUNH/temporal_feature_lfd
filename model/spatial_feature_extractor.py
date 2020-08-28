import torch
import torch.nn as nn

from .backbone_model.tsm.ops.basic_ops import ConsensusModule

class SpatialFeatureExtractor(nn.Module):   
	def __init__(self, lfd_params, is_training ):

		super().__init__()

		self.lfd_params = lfd_params 

		self.num_classes = lfd_params.num_actions
		self.use_aud =  lfd_params.use_aud
		self.is_training = is_training
		self.num_segments =  lfd_params.args.num_segments

		self.bottleneck_size = lfd_params.args.bottleneck_size

		self.checkpoint_file = lfd_params.args.pretrain_modelname
		train_backbone = self.is_training


		if (lfd_params.args.cnn_modelname):
			self.checkpoint_file = lfd_params.args.cnn_modelname
			train_backbone = False



		'''
		Need to stop gradient from backpropagating through model
			torch.no_grad()

		Need to save specific features and not others 
			--policy_modelname

		'''


		# rgb net
		from .backbone_model.tsm.tsm import TSMWrapper as VisualFeatureExtractor
		self.rgb_net = VisualFeatureExtractor(
			self.checkpoint_file,
			self.num_classes, 
			num_segments=self.num_segments
			)

		# prevent the training of these layers by removing their grad information
		if (train_backbone):
			for param in self.rgb_net.parameters():
				param.requires_grad = False

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

		print("self.rgb_net.state_dict():")
		for k in self.rgb_net.state_dict().keys():
			print("\t"+k, self.rgb_net.state_dict()[k].shape )

		print("linear.state_dict():")
		for k in self.linear.state_dict().keys():
			print("\t"+k, self.linear.state_dict()[k].shape )

	# Defining the forward pass    
	def forward(self, rgb_x):

		# pass data through CNNs
		rgb_y = self.rgb_net(rgb_x)

		# apply linear layer and consensus module to the output of the CNN
		if (self.is_training):
			rgb_y = rgb_y.view((-1, self.rgb_net.num_segments) + rgb_y.size()[1:])
		else:
			rgb_y = rgb_y.view((-1, self.rgb_net.num_segments*10) + rgb_y.size()[1:])

		rgb_y = self.consensus(rgb_y)
		rgb_y = rgb_y.squeeze(1)

		obs_y = self.linear(rgb_y)

		return obs_y

		# if using audio data as well I need to combine those features
		'''
		if (self.use_aud):
			aud_y = self.aud_net(aud_x)
			obs_x = torch.stack([rgb_y, aud_y], dim=0, out=None)
		else:
			obs_x = rgb_y
		'''

	def save_model(self):
		pass