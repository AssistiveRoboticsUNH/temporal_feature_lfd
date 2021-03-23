import os
import torch
import torch.nn as nn


class SpatialBottleneck(nn.Module):
	def __init__(self, lfd_params, is_training=False, filename=None,
				 input_size=2048, bottleneck_size=128, spatial_size=7, resize_output=True):
		super().__init__()
		self.lfd_params = lfd_params

		# model filenames
		self.filename = os.path.join(filename, ".".join(["model", "spatial_bottleneck", "pt"]))

		# constants params
		self.input_size = input_size
		self.bottleneck_size = bottleneck_size
		self.spatial_size = spatial_size

		self.resize_output = resize_output

		# define model vars
		if self.resize_output:
			self.bottleneck = nn.Sequential(
				nn.Conv2d(self.input_size, self.bottleneck_size, (1, 1)),
				nn.AdaptiveMaxPool2d(output_size=1),
			)
		else:
			self.bottleneck = nn.Sequential(
				nn.Conv2d(self.input_size, self.bottleneck_size, (1, 1)),
			)

		# load model parameters
		if not is_training:
			assert self.filename is not None, \
				"ERROR: spatial_bottleneck.py: filename must be defined when is_training is False"
			self.load_model(self.filename)#, self.bottleneck)
		else:
			print("SpatialBottleneck is training")

	# Defining the forward pass
	def forward(self, x):
		x = x.view(-1, self.input_size, self.spatial_size, self.spatial_size)  # I3D
		x = self.bottleneck(x)

		if self.resize_output:
			x = x.view(self.lfd_params.batch_size, -1, self.bottleneck_size)
		return x

	def save_model(self):
		torch.save(self.state_dict(), self.filename)
		print("SpatialBottleneck model saved to: ", self.filename)

	def load_model(self, filename):
		assert os.path.exists(filename), "ERROR: spatial_bottleneck.py: Cannot locate saved model - " + filename

		print("Loading SpatialBottleneck from: " + filename)
		checkpoint = torch.load(filename)
		self.load_state_dict(checkpoint, strict=True)
		for param in self.parameters():
			param.requires_grad = False

	''' 
	def save_model(self, filename):
		torch.save(self.bottleneck.state_dict(), filename)
		print("SpatialBottleneck Conv2d model saved to: ", filename)

	def load_model(self, filename, var):
		assert os.path.exists(filename), "ERROR: spatial_bottleneck.py: Cannot locate saved model - " + filename

		print("Loading SpatialBottleneck from: " + filename)
		checkpoint = torch.load(filename)
		var.load_state_dict(checkpoint, strict=True)
		for param in var.parameters():
			param.requires_grad = False
	'''