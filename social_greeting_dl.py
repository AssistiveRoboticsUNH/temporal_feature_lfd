import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os, sys

#sys.path.insert(0, "/home/mbc2004/temporal-shift-module")
#from ops.transforms import * 
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
	def __init__(self, root_path, transform, mode, segment_length=8, num_segments=3, image_tmpl=IMAGE_TMPL_DEF):

		assert mode in ["train", "test"], "ERROR: Mode param must be 'train' or 'test'"
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
		self.transform = transform

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
		start_idx = self.get_indexes(filename)

		# collect array of frames into list
		images = []
		for idx in range(1, len(self.segment_length)+1):
			images.extend( [Image.open(os.path.join(filename, self.image_tmpl.format(start_idx + idx))).convert('RGB')] )

		# return the processed images 
		return self.transform(images)

	def get_indexes(self, filename):

		total_num_frames = len(os.listdir(filename))
		if self.mode == "train":
			# get random indexes
			return np.random.randint(0, total_num_frames-self.segment_length, 1)
		else:
			# get dense sampling	
			print("DENSE SAMPLING NOT IDEAL")
			return np.linspace(0, total_num_frames-self.segment_length, num=10, dtype=int)	

	def __len__(self):
		return len(self.data)

class SocialGreetingDataSet(VideoDataset):

	class Data:
		def __init__(self, filename, action, history):
			self.filename = filename
			self.history = history
			self.action = action #label

	def __init__(self, root_path, transform, image_tmpl=IMAGE_TMPL_DEF):

		super().__init__(root_path, transform, image_tmpl=image_tmpl)

		self.action_dict = {
			'g':  [1],
			'a':  [1],
			'ga': [1],
			'za': [1],
			'zg': [1],
			'zga':[1],
			'z':    [0, 2],
			'none': [0, 2]}

		self.history = {
			0:[0],
			1:[0, 1],
			2:[1]
		}

		# generate all observation, hidden state, action combinations
		self.data = []

		print("obs:", self.obs_dict.keys())

		for obs_category in self.obs_dict.keys():
			for obs_sample in self.obs_dict[obs_category]:

				obs_file_dir = os.path.join(*[root_path, obs_category, obs_sample])

				for action in self.action_dict[obs_category]:
					for history in self.history[action]:
						self.data.append( self.Data(obs_file_dir, history, action) )
		
	def __getitem__(self, index):

		data = self.data[index]

		obs_x = self.parse_obs(data.filename)
		world_x = data.history
		action_y = data.action

		#print(type(obs_x), type(world_x), type(action_y))
		print(index, obs_x.size(), world_x, action_y)

		return obs_x, world_x, action_y


def create_dataloader(file_path, mode, batch_size=1, num_workers=16, max_length=8):

	# define transform function
	transform = torchvision.transforms.Compose([
		torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66])]),
		#np.concatenate(img_group, axis=2)
		Stack(roll=(False)), # this is the culprit
		ToTorchFormatTensor(div=(True)),
		IdentityTransform(),
		])

	# setup path parameters
	assert mode in ["train", "validate", "evaluate"], "ERROR: mode must be either 'train', 'validate', or 'evaluate'"

	root_path = os.path.join(file_path, mode)

	shuffle = False
	if mode == "train":
		shuffle = True

	# create dataset
	dataset = SocialGreetingDataSet( root_path,
		image_tmpl='image_{:05d}.jpg',
		transform=transform,
		segment_length=max_length )

	# create dataloader
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers, 
		pin_memory = True)
