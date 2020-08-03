import torch
import os, sys

sys.path.append("/home/mbc2004/temporal-shift-module/ops")
#from ops.dataset import TSNDataSet
from transforms import * 

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

I need to get this working with an example ()use Something SOmething for now. So that 
I can make sure the rest of the application works. 

'''

IMAGE_TMPL_DEF = '{:05d}.jpg'

class VideoDataset(data.Dataset):
	def __init__(self, root_path, transform, image_tmpl=IMAGE_TMPL_DEF):

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
		full_path = os.path.join(self.root_path, record.path, self.image_tmpl.format(1))
		assert os.path.exists(full_path), 'ERROR: Directory Not Found - '+full_path
		assert len(os.listdir(full_path)) > 0, 'ERROR: Directory Empty - '+full_path

		# collect array of frames into list
		images = []
		for idx in range(len(os.listdir(full_path))):
			images.extend( [Image.open(os.path.join(full_path, self.image_tmpl.format(idx))).convert('RGB')] )

		# return the processed images 
		return self.transform(images)

	def __len__(self):
		return len(self.data)

class SocialGreetingDataSet(VideoDataset):

	class Data:
		def __init__(self, filename, action, history):
			self.filename = filename
			self.history = history
			self.action = action #label

	def __init__(self, root_path, transform, image_tmpl=IMAGE_TMPL_DEF):

		super(SocialGreetingDataSet).__init__(self, root_path, transform, image_tmpl)

		self.action_dict = {
			'g':  [1],
			'a':  [1],
			'ga': [1],
			'za': [1],
			'gza':[1],
			'z':    [0, 2],
			'none': [0, 2]}

		self.history = {
			0:[0],
			1:[0, 1],
			2:[1]
		}

		# generate all observation, hidden state, action combinations
		self.data = []

		for obs_category in obs_dict.keys():
			for obs_sample in obs_dict[obs_category]:

				obs_file_dir = os.path.join([root_path, obs_category, obs_sample])

				for action in self.action_dict[obs_root]:
					for history in self.history[action]:
						self.data.append( Data(obs_file_dir, history, action) )
		
	def __getitem__(self, index):
		data = self.data[index]

		obs_x = self.parse_obs(data.filename)
		world_x = data.history
		action_y = data.action

		return obs_x, world_x, action_y


def create_dataloader(file_path, mode, batch_size=8, num_workers=16):

	# define transform function
	transform = torchvision.transforms.Compose([
		torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66])]),
		Stack(roll=(False)),
		ToTorchFormatTensor(div=(True)),
		IdentityTransform(),
		])

	# setup path parameters
	assert mode in ["train", "validate", "evaluate"], "ERROR: mode must be either 'train', 'validate', or 'evaluate'"

	root_path = os.path.join(params.obs_file_path, mode)

	shuffle = False
	if mode == "train":
		shuffle = True

	# create dataset
	dataset = SocialGreetingDataSet( root_path,
		image_tmpl='{:06d}.jpg',
		transform=transform )

	# create dataloader
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers, 
		pin_memory = True)
