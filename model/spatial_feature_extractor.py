import torch
import torch.nn as nn

from .backbone_model.tsm.ops.basic_ops import ConsensusModule
from .feature_extractor import FeatureExtractor


class SpatialFeatureExtractor(FeatureExtractor):

	def __init__(self, lfd_params, is_training ):
		super().__init__(lfd_params, is_training)

		# define an extension layer that takes the output of the backbone and obtains 
		# action labels from it.
		self.linear_dimension = self.bottleneck_size * self.num_segments
		self.linear = nn.Sequential(
			nn.Linear(self.linear_dimension, self.num_classes)
		)
		self.consensus = ConsensusModule('avg')

		ext_checkpoint = self.lfd_params.args.ext_modelname
		if (ext_checkpoint):

			# load saved model parameters	
			print("spatial_feature_extractor.py: Loading Extension Model from: ", ext_checkpoint)	
			checkpoint = torch.load(ext_checkpoint)
			self.linear.load_state_dict(checkpoint, strict=True)

			# prevent changes to these parameters
			for param in self.linear.parameters():
				param.requires_grad = False	
		else:
			print("spatial_feature_extractor.py: Did Not Load Extension Model")	

	def forward(self, rgb_x):

		# pass data through CNNs
		rgb_y = self.rgb_net(rgb_x)

		# apply linear layer and consensus module to the output of the CNN
		#rgb_y = rgb_y.view((-1, self.rgb_net.num_segments) + rgb_y.size()[1:])
		rgb_y = rgb_y.view((-1, self.rgb_net.num_segments * self.bottleneck_size))
		print("rgb_y:", rgb_y.shape)
		#rgb_y = self.consensus(rgb_y)
		#rgb_y = rgb_y.squeeze(1)
		obs_y = self.linear(rgb_y)

		return obs_y

	def save_model(self):
		super().save_model()

		filename = self.lfd_params.generate_ext_modelname()
		torch.save(self.linear.state_dict(), filename)
		print("Ext model saved to: ", filename)
