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
		input_size = 224
		if (self.mode == "train"):
			self.transform = torchvision.transforms.Compose([
				torchvision.transforms.Compose([
					GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
					GroupRandomHorizontalFlip(is_flow=False)]),
				Stack(roll=(False)), 
				ToTorchFormatTensor(div=(True)),
				IdentityTransform(),
				])
		else:
			self.transform = torchvision.transforms.Compose([
				GroupScale(224),#input_size * 256 // 224),
                GroupCenterCrop(256),
				Stack(roll=(False)), 
				ToTorchFormatTensor(div=(True)),
				IdentityTransform(),
				])

		self.segment_length = segment_length
		self.num_segments = num_segments

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

		# sample images
		images = self.regular_sampling(filename)

		# return the processed images 
		return self.transform(images)

	def regular_sampling(self, filename):
		# get start indexes of frames
		total_num_frames = len(os.listdir(filename))
		start_indexes = self.get_start_indexes(total_num_frames)
		stride = 64 // self.num_segments

		# collect array of frames into list
		images = []
		for start_idx in start_indexes:
			frame_indexes = [(idx * stride + start_idx) % total_num_frames for idx in range(self.num_segments)]
			print(frame_indexes)

			images.extend( [Image.open(os.path.join(filename, self.image_tmpl.format(start_idx + idx))).convert('RGB') for idx in frame_indexes ] )
		return images

	def get_start_indexes(self, total_num_frames):
		
		if self.mode == "train":
			# get random indexes
			return np.random.randint(0, total_num_frames-self.segment_length, self.num_segments)
		else:
			# get dense sampling	
			return np.linspace(0, total_num_frames-self.segment_length, num=10, dtype=int)

	def __len__(self):
		return len(self.data)

