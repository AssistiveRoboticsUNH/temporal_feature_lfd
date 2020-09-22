import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):

	def __init__(self, lfd_params, is_training ):

		super().__init__()

		self.lfd_params = lfd_params 

		self.num_classes = lfd_params.num_actions
		self.use_aud = lfd_params.use_aud
		self.num_segments = lfd_params.args.num_segments
		self.is_training = is_training

		self.bottleneck_size = lfd_params.args.bottleneck_size

		# RGB MODEL 
		# ---

		# get the files to use with this model
		self.checkpoint_file = lfd_params.args.pretrain_modelname
		pretrained_checkpoint = True
		if lfd_params.args.backbone_modelname:
			self.checkpoint_file = lfd_params.args.backbone_modelname
			pretrained_checkpoint = False

		# rgb net

		# TSM
		from .backbone_model.tsm.tsm import TSMWrapper as VisualFeatureExtractor

		# SimpleNet
		#from .backbone_model.simplenet import SimpleNet as VisualFeatureExtractor

		self.rgb_net = VisualFeatureExtractor(
			self.checkpoint_file,
			self.num_classes, 
			training=self.is_training,
			num_segments=self.num_segments,
			pretrained_checkpoint=pretrained_checkpoint,
			bottleneck_size=self.bottleneck_size
			)

		# parameter indicates that the backbone's features should be fixed
		# the following code prevents the modification of these layers by removing their gradient information
		if lfd_params.args.backbone_modelname:
			for param in self.rgb_net.parameters():
				param.requires_grad = False

	# Defining the forward pass    
	def forward(self, rgb_x):
		return self.rgb_net(rgb_x)

	def save_model(self):
		filename = self.lfd_params.generate_backbone_modelname()
		self.rgb_net.save_model(filename)
		print("Backbone model saved to: ", filename)
