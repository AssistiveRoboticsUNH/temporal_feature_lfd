import torch
import torch.nn as nn

from .backbone_model.tsm.ops.basic_ops import ConsensusModule
from .feature_extractor import FeatureExtractor

class SpatialFeatureExtractor(FeatureExtractor):   
	def __init__(self, lfd_params, is_training ):
		super().__init__(lfd_params, is_training)

		# define an extension layer that takes the output of the backbone and obtains 
		# action labels from it.
		self.linear_dimension = self.bottleneck_size
		self.linear = nn.Sequential(
			nn.Linear(self.linear_dimension, self.num_classes)
		)
		self.consensus = ConsensusModule('avg')

		ext_checkpoint = self.lfd_params.args.ext_modelname
		if (ext_checkpoint):

			# load saved model parameters		
			checkpoint = torch.load(ext_checkpoint)['state_dict']
			self.linear.load_state_dict(checkpoint, strict=False)

			# prevent changes to these parameters
			for param in self.linear.parameters():
				param.requires_grad = False	

	# Defining the forward pass    
	def forward(self, rgb_x):

		# pass data through CNNs
		rgb_y = self.rgb_net(rgb_x)

		# apply linear layer and consensus module to the output of the CNN
		if (self.is_training):
			rgb_y = rgb_y.view((-1, self.rgb_net.num_segments) + rgb_y.size()[1:])
		else:
			rgb_y = rgb_y.view((-1, self.rgb_net.num_segments*10) + rgb_y.size()[1:])

		# pass through spatial extension
		# ---

		rgb_y = self.consensus(rgb_y)
		rgb_y = rgb_y.squeeze(1)

		obs_y = self.linear(rgb_y)

		return obs_y

	def save_model(self, debug=False):
		super().save_model(debug)

		if (debug):
			print("linear.state_dict():")
			for k in self.linear.state_dict().keys():
				print("\t"+k, self.linear.state_dict()[k].shape )

		torch.save(self.linear.state_dict(),  self.lfd_params.generate_ext_modelname() )