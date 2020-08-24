import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os, sys

from model.backbone_model.tsm.ops.transforms import *

'''
File structure is:
  /root
	/train
	/validate
	/eval
	  /obs1
	  /obs2
	  /obs3
	  	/example1
	  	/example2
	  	/example3
			/00000.jpg frame1
			/00001.jpg frame2
			/00002.jpg frame3

Here is what I need to do: The dataset loader needs to open the video file and output 
the video, state history, and the action label. 

I need to get this working with an example () use Something SOmething for now. So that 
I can make sure the rest of the application works. 

'''

IMAGE_TMPL_DEF = '{:05d}.jpg'

class VideoDataset(Dataset):
	def __init__(self, root_path, mode, segment_length=8, num_segments=3, image_tmpl=IMAGE_TMPL_DEF, flip=False):

		assert mode in ["train", "evaluation"], "ERROR: Mode param must be 'train' or 'evaluation'"
		self.mode = mode

		assert os.path.exists(root_path), "ERROR: Cannot locate path - "+root_path
		self.root_path = root_path

		# get the video files
		self.data = []
		self.obs_dict = {}

		for obs in os.listdir(root_path):
			all_obs_files = os.listdir(os.path.join(root_path, obs))
			self.obs_dict[obs] = all_obs_files
			self.data.extend(all_obs_files)

		# how to transform the images
		if (self.mode == "train"):
			self.transform = torchvision.transforms.Compose([
				torchvision.transforms.Compose([
					GroupMultiScaleCrop(224, [1, .875, .75, .66]),
					GroupRandomHorizontalFlip(is_flow=False)]),
				Stack(roll=(False)), 
				ToTorchFormatTensor(div=(True)),
				IdentityTransform(),
				])
		else:
			self.transform = torchvision.transforms.Compose([
				GroupScale(256 // 224),
                GroupCenterCrop(224),
				Stack(roll=(False)), 
				ToTorchFormatTensor(div=(True)),
				IdentityTransform(),
				])

		self.segment_length = segment_length

		# template for how the filename is stored
		self.image_tmpl = image_tmpl 

	def __getitem__(self, index):
		filename = self.data[index]
		obs_x = self.parse_obs(filename)
		return obs_x

	def parse_obs(self, filename):

		# check this is a legit video folder
		assert os.path.exists(filename), 'ERROR: Directory Not Found - '+filename
		assert len(os.listdir(filename)) > 0, 'ERROR: Directory Empty - '+filename

		# get start indexes of frames
		start_idx = self.get_indexes(filename)[0]

		# collect array of frames into list
		images = []
		for idx in range(1, self.segment_length+1):
			images.extend( [Image.open(os.path.join(filename, self.image_tmpl.format(start_idx + idx))).convert('RGB')] )

		# return the processed images 
		return self.transform(images)

	def get_indexes(self, filename):

		total_num_frames = len(os.listdir(filename))
		print("self.get_indexes", self.mode)
		if self.mode == "train":
			# get random indexes
			return np.random.randint(0, total_num_frames-self.segment_length, 1)
		else:
			idx = np.linspace(0, total_num_frames-self.segment_length, num=10, dtype=int)
			print("self.get_indexes", self.mode, len(idx), total_num_frames-self.segment_length)
			# get dense sampling	
			return idx

	def __len__(self):
		return len(self.data)

