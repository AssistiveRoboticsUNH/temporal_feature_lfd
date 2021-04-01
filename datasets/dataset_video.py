from model.backbone_model.tsm.ops.transforms import *
import numpy as np
import os
from PIL import ImageFilter, Image
from torch.utils.data import Dataset, DataLoader

from .utils import get_observation_list
from iad_analyze import convert_iad_to_sparse_map

# need to remove ros path before I can import cv2
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

IMAGE_TMPL_DEF = 'image_{:05d}.jpg'


class DifferenceMask(object):
    def __init__(self, gaussian_value=1, kernel_size=(3, 3)):
        self.gaussian_value = gaussian_value
        self.kernel_size = kernel_size

    def __call__(self, img_group):
        # prepare back subtraction methods
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        back_subtract = cv2.createBackgroundSubtractorMOG2()

        # apply background removal to first frame of video
        frame = np.array(img_group[0].filter(ImageFilter.GaussianBlur(self.gaussian_value)))
        _ = back_subtract.apply(frame)

        image_out = []
        for img in img_group[1:]:
            # smooth video to reduce slight variances
            frame = np.array(img.filter(ImageFilter.GaussianBlur(self.gaussian_value)))

            fg_mask = back_subtract.apply(frame)  # remove background
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # filter noise in mask
            fg_mask[fg_mask > 0] = 255
            fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0  # convert to float values

            img_f = (fg_mask * frame).astype(np.uint8)  # apply mask to frame
            img_f = Image.fromarray(img_f)  # convert frame to PIL image
            image_out.append(img_f)

        image_out.append(Image.fromarray(np.zeros_like(img_group[0])))  # pad data to required length

        return image_out


class DatasetVideo(Dataset):
    def __init__(self, lfd_params, root_path, mode, verbose=False, dataset_mode=None,
                 image_tmpl=IMAGE_TMPL_DEF, num_segments=3, backbone="",
                 specific_labels=None):

        assert mode in ["train", "evaluation"], "ERROR: dataset_video.py: Mode param must be 'train' or 'evaluation'"
        self.mode = mode
        self.verbose = verbose
        self.dense_sample = False #lfd_params.dense_sample
        self.dense_rate = 1 #lfd_params.dense_rate
        self.specific_labels = specific_labels

        if dataset_mode is None:
            dataset_mode = mode

        root_path = os.path.join(root_path, "frames")
        assert os.path.exists(root_path), "ERROR: dataset_video.py: Cannot locate path - " + root_path
        self.obs_dict = get_observation_list(lfd_params, root_path, dataset_mode)

        # make data easily accessible
        self.data = []
        for k in self.obs_dict:
            self.data.extend(self.obs_dict[k])

        # define dataset params
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.obs_label_list = lfd_params.application.obs_label_list  #{"n": 0, "r": 1, "rr": 2, "rrr": 3, "g": 4, "gb": 5, "bg": 6, "b": 7}

        # define transform function
        input_size = 224
        if self.mode == "train":
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Compose([
                    GroupScale(224),#
                    GroupCenterCrop(224),#

                    #GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                    DifferenceMask(),
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

    def __getitem__(self, index):
        filename = self.data[index]
        label = self.get_label(filename)

        if self.specific_labels is not None and label not in self.specific_labels:
            obs = None
        else:
            obs = self.parse_obs(filename)

        if self.verbose:
            return obs, label, filename
        return obs, label

    def __len__(self):
        return len(self.data)

    def parse_obs(self, filename):

        # check this is a legitimate video folder
        assert os.path.exists(filename), 'ERROR: Directory Not Found - ' + filename
        assert len(os.listdir(filename)) > 0, 'ERROR: Directory Empty - ' + filename

        # get start indexes of frames
        total_num_frames = len(os.listdir(filename))

        dense_num = self.dense_rate

        if self.dense_sample:
            start_idxs = np.linspace(0, max(1, total_num_frames - self.num_segments), num=dense_num, dtype=int) + 1

            images = [Image.open(os.path.join(filename, self.image_tmpl.format(idx))).convert('RGB') for s_idx in start_idxs for idx in range(s_idx, s_idx+self.num_segments)]

        else:
            idxs = np.linspace(0, max(1, total_num_frames - 1), num=self.num_segments, dtype=int) + 1

                # collect array of frames into list
            images = [Image.open(os.path.join(filename, self.image_tmpl.format(idx))).convert('RGB') for idx in idxs]

        # transform the images using the defined model
        images = self.transform(images)

        # reshape the images to work with model
        if self.dense_sample:
            images = torch.reshape(images, (dense_num, -1, self.num_segments * 3, 224, 224))
        else:
            images = torch.reshape(images, (-1, self.num_segments * 3, 224, 224))

        return images

    def get_label(self, filename):
        obs_name = filename.split('/')[-2]
        return self.obs_label_list[obs_name]

    def show(self, index):
        if self.verbose:
            img, _, _ = self.__getitem__(index)
        else:
            img, _ = self.__getitem__(index)

        img = img.numpy()
        img = np.reshape(img, (-1, 3, img.shape[2], img.shape[3]))
        img = np.transpose(img, (0, 2, 3, 1))
        img *= 255
        img = img.astype(np.uint8)

        def get_concat_h(im1, im2):
            buffer = 0  # 5
            dst = Image.new('RGB', (im1.width + im2.width + buffer, im1.height), (255, 255, 255))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width+buffer, 0))
            return dst

        img_all = Image.fromarray(img[0])
        for img_f in img[1:]:
            img_f = Image.fromarray(img_f)
            img_all = get_concat_h(img_all, img_f)
        return img_all
