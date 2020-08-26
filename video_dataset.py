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
	'''
			root_path, 						# str; the path to the directory containing the videos as images
			mode, 							# str; the mode to run the model 'train' or 'evaluation'
			num_segments=3, 				# int; the number of segments in the clip
			image_tmpl=IMAGE_TMPL_DEF, 		# str; the template for the file name, can be formatted with an integer 
			flip=False,						# bool; flip some of the vidoes horizontally
			clip_size=64					# int; the length of the clipfrom which to sample frames
	'''


	def __init__(self, 
			root_path, 
			mode, 
			num_segments=3, 
			image_tmpl=IMAGE_TMPL_DEF, 
			flip=False,	
			clip_size=64
		):

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
                GroupCenterCrop(224),#256),
				Stack(roll=(False)), 
				ToTorchFormatTensor(div=(True)),
				IdentityTransform(),
				])

		self.num_segments = num_segments
		self.image_tmpl = image_tmpl 
		self.clip_size = clip_size

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
		images = self.transform(images)
		images = torch.reshape(images, (-1, self.num_segments, 3, 224,224))
		return images

	def regular_sampling(self, filename):
		# get start indexes of frames
		total_num_frames = len(os.listdir(filename))
		start_indexes = self.get_start_indexes(total_num_frames)
		stride = self.clip_size // self.num_segments

		# collect array of frames into list
		images = []
		for start_idx in start_indexes:
			frame_indexes = np.array([(idx * stride + start_idx) % (total_num_frames-1) for idx in range(self.num_segments)])+1
			images.extend( [Image.open(os.path.join(filename, self.image_tmpl.format(idx))).convert('RGB') for idx in frame_indexes ] )
		return images

	def get_start_indexes(self, total_num_frames):
		
		if self.mode == "train":
			# get random indexes
			return np.random.randint(0, max(1, 1 + total_num_frames - self.clip_size), 1)
		else:
			# get dense sampling	
			return np.linspace(0, max(1, 1 + total_num_frames - self.clip_size), num=10, dtype=int)

	def __len__(self):
		return len(self.data)

