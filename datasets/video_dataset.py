from torch.utils.data import Dataset
from model.backbone_model.tsm.ops.transforms import *

import numpy as np
import os

from PIL import ImageFilter, Image

# need to remove ros path before I can import cv2
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

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

"""
def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image, (65, 65), 10)
    new_image = image_blur
    return new_image
"""


class GaussianBlur(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, gaussian_value=0):
        self.gaussian_value = gaussian_value

    def __call__(self, img_group, is_flow=False):
        out_group = []
        for img in img_group:
            out_group.append(img.filter(ImageFilter.GaussianBlur(self.gaussian_value)))

        return out_group


class DifferenceMask(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, gaussian_value=1, kernel_size=(3, 3)):
        self.gaussian_value = gaussian_value
        self.kernel_size = kernel_size

    def __call__(self, img_group, is_flow=False):
        # convert to gray scale
        image_out = []

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        back_subtractor = cv2.createBackgroundSubtractorMOG2()

        # apply background removal to first frame of video
        frame = np.array(img_group[0].filter(ImageFilter.GaussianBlur(self.gaussian_value)))
        _ = back_subtractor.apply(frame)

        for img in img_group[1:]:
            # smooth video to reduce slight variances
            frame = np.array(img.filter(ImageFilter.GaussianBlur(self.gaussian_value)))

            fg_mask = back_subtractor.apply(frame)  # remove background
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # filter noise in mask
            fg_mask[fg_mask > 0] = 255
            fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0  # convert to float values

            img_f = (fg_mask * frame).astype(np.uint8)  # apply mask to frame
            img_f = Image.fromarray(img_f)  # convert frame to PIL image
            image_out.append(img_f)

        image_out.append(Image.fromarray(np.zeros_like(img_group[0])))  # pad data to required length

        return image_out


class VideoDataset(Dataset):
    """
    root_path, 						# str; the path to the directory containing the videos as images
    mode, 							# str; the mode to run the model 'train' or 'evaluation'
    num_segments=3, 				# int; the number of segments in the clip
    image_tmpl=IMAGE_TMPL_DEF, 		# str; the template for the file name, can be formatted with an integer
    flip=False,						# bool; flip some of the vidoes horizontally
    clip_size=64					# int; the length of the clipfrom which to sample frames
    """

    def __init__(self,
                 root_path,
                 mode,
                 full_sample,
                 num_segments=3,
                 image_tmpl=IMAGE_TMPL_DEF,
                 clip_size=64,
                 fix_stride=1,
                 gaussian_value=0
                 ):

        assert mode in ["train", "evaluation"], "ERROR: Mode param must be 'train' or 'evaluation'"
        self.mode = mode

        assert os.path.exists(root_path), "ERROR: Cannot locate path - "+root_path
        self.root_path = root_path

        # get the video files
        self.data = []
        self.obs_dict = {}
        self.get_filename = False

        for obs in os.listdir(root_path):
            #if obs in ['r', 'g', 'b']:
            all_obs_files = os.listdir(os.path.join(root_path, obs))
            self.obs_dict[obs] = all_obs_files
            for obs_files in all_obs_files:
                self.data.append(os.path.join(*[root_path, obs, obs_files]))

        # how to transform the images
        input_size = 224
        if self.mode == "train":
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Compose([
                    GroupScale(224),
                    GroupCenterCrop(224),
                    #GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                    DifferenceMask(),
                    #GaussianBlur(gaussian_value),
                    #GroupRandomHorizontalFlip(is_flow=False)
                    ]),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                IdentityTransform(),
                ])
        else:
            self.transform = torchvision.transforms.Compose([
                GroupScale(224),
                GroupCenterCrop(224),
                DifferenceMask(),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                IdentityTransform(),
                ])

        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.clip_size = clip_size

        self.full_sample = full_sample
        self.fix_stride = fix_stride

    def show(self, index):
        if (self.get_filename):
            img, _ = self.__getitem__(index)
        else:
            img = self.__getitem__(index)

        img = img.numpy()
        img = np.reshape(img, (-1, 3, img.shape[2], img.shape[3]))
        img = np.transpose(img, (0, 2, 3, 1))
        img *= 255
        img = img.astype(np.uint8)

        def get_concat_h(im1, im2):
            dst = Image.new('RGB', (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))
            return dst

        img_all = Image.fromarray(img[0])
        for img_f in img[1:]:
            img_f = Image.fromarray(img_f)
            img_all = get_concat_h(img_all, img_f)
        return img_all

    def __getitem__(self, index):
        filename = self.data[index]
        obs_x = self.parse_obs(filename)
        obs_x = torch.reshape(obs_x, (-1, self.num_segments * 3, 224, 224))

        print("self.get_filename:", self.get_filename)

        if self.get_filename:
            return obs_x, filename
        return obs_x

    def parse_obs(self, filename):

        # check this is a legit video folder
        assert os.path.exists(filename), 'ERROR: Directory Not Found - '+filename
        assert len(os.listdir(filename)) > 0, 'ERROR: Directory Empty - '+filename

        # sample images
        #if (not self.full_sample):
        images = self.regular_sampling(filename)
        #else:
        #	images = self.dense_sampling(filename)

        # return the processed images
        images = self.transform(images)
        return images

    def regular_sampling(self, filename):
        # get start indexes of frames
        total_num_frames = len(os.listdir(filename))
        start_indexes = self.get_start_indexes(total_num_frames)
        stride = self.clip_size // self.num_segments

        # collect array of frames into list
        images = []

        #if (self.mode == "train"):
        idxs = np.linspace(0, max(1, total_num_frames-1), num=self.num_segments, dtype=int)+1
        #else:
        #	idxs = np.linspace(0, max(1, total_num_frames-1), num=self.num_segments*10, dtype=int)+1

        for idx in idxs:
            images.append( Image.open(os.path.join(filename, self.image_tmpl.format(idx))).convert('RGB') )

        '''
        for start_idx in start_indexes:
            frame_indexes = np.array([(idx * stride + start_idx) % (total_num_frames-1) for idx in range(self.num_segments)])+1
            images.extend( [Image.open(os.path.join(filename, self.image_tmpl.format(idx))).convert('RGB') for idx in frame_indexes ] )
        '''
        return images

    def get_start_indexes(self, total_num_frames):

        if self.mode == "train" and not self.full_sample:
            # get random indexes
            return np.random.randint(0, max(1, 1 + total_num_frames - self.clip_size), 1)
        else:
            # get dense sampling
            return np.linspace(0, max(1, 1 + total_num_frames - self.clip_size), num=10, dtype=int)

    def dense_sampling(self, filename):
        # get start indexes of frames
        total_num_frames = len(os.listdir(filename))

        # collect array of frames into list
        images = [Image.open(os.path.join(filename, self.image_tmpl.format(idx))).convert('RGB') for idx in range(1, total_num_frames, self.fix_stride) ]
        return images

    def __len__(self):
        return len(self.data)


class ITRDataset(Dataset):
    def __init__(self, root_path, mode):
        assert mode in ["train", "evaluation"], "ERROR: Mode param must be 'train' or 'evaluation'"
        self.mode = mode

        assert os.path.exists(root_path), "ERROR: Cannot locate path - " + root_path
        self.root_path = root_path

        # get the ITR files
        self.data = []
        self.obs_dict = {}

        for obs in os.listdir(root_path):
            all_obs_files = os.listdir(os.path.join(root_path, obs))
            self.obs_dict[obs] = all_obs_files
            self.data.extend(all_obs_files)

    def parse_obs(self, filename):
        return np.load(filename)["data"]

    def __getitem__(self, index):
        itr_filename = self.data[index]
        return self.parse_obs(itr_filename)

    def __len__(self):
        return len(self.data)
