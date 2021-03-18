#import cv2

# pass video through conv net.
# Find out where the features are prior to the max function
# scale and display.

import numpy as np
import os
import copy
import PIL
from PIL import Image

from enums import suffix_dict, model_dict, Suffix
from parameter_parser import default_model_params

from model.classifier import Classifier
from execute import make_model_name
from datasets.dataset_video import DatasetVideo

import torch
from scipy.signal import savgol_filter


def save_png(img, output_filename, swap_color=False):





def convert_to_img(args, rgb_img, activation_map):
    rgb_img = rgb_img.reshape([args.frames, rgb_img.shape[-2], rgb_img.shape[-1], 3])
    rgb_img *= 255
    rgb_img = rgb_img.astype(np.uint8)

    activation_map = activation_map.transpose([1, 0, 2, 3])

    print("rgb_img.shape:", rgb_img.shape)
    print("activation_map.shape:", activation_map.shape)

    num_frames, height, width = rgb_img.shape[0], rgb_img.shape[1], rgb_img.shape[2]

    dst = Image.new('RGB', (width * num_frames, height))
    for t in range(num_frames):
        #print("rgb_img[t]:", rgb_img[t].shape, rgb_img[0][0])

        img_frame = Image.fromarray(rgb_img[t])
        #activation_frame = Image.fromarray(activation_map[t]).resize((width, height), PIL.Image.NEAREST)

        #print("img_frame:", img_frame.shape)
        #print("dst:", dst.shape)


        dst.paste(img_frame, (width * t, 0))
        #img = np.concatenate([img, rgb_img[t]], axis=1)

    return dst
    #print("dst.szie:", dst.size)


def exec_func(args, lfd_params):

    # define datasets
    train_files = DatasetVideo(lfd_params, lfd_params.application.file_directory, "train", verbose=True,
                               num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
    evaluation_files = DatasetVideo(lfd_params, lfd_params.application.file_directory, "evaluation", verbose=True,
                                    num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)

    # define model
    # filename = make_model_name(args, lfd_params, backbone=False)
    model = Classifier(lfd_params, args.filename, model_dict[args.model], Suffix.NONE,
                       use_feature_extractor=True, train_feature_extractor=False,
                       use_bottleneck=True,
                       use_spatial=False, train_spatial=False,
                       use_pipeline=False, train_pipeline=False,
                       use_temporal=False, train_temporal=False,

                       resize_bottleneck=False)

    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    # generate feature presence
    for dataset_files in [train_files, evaluation_files]:
        for obs, label, filename in dataset_files:

            # compute output
            activation_map = net(obs)
            obs = obs.detach().cpu().numpy()
            activation_map = activation_map.detach().cpu().numpy()
            print(activation_map.shape)

            img_out = convert_to_img(args, obs, activation_map)

            filename_split = filename.split('/')
            filename_id = filename_split[-1].split('.')[0] + ".png"
            obs_id = filename_split[-2]
            mode_id = filename_split[-3]
            png_dir = os.path.join(*[lfd_params.application.file_directory, "intr_png", mode_id, obs_id])
            if not os.path.exists(png_dir):
                os.makedirs(png_dir)
            output_filename = os.path.join(png_dir, filename_id)

            img_out.save(output_filename, "PNG")


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize where features occur in the dataset')

    parser.add_argument('model', help='model_id', choices=model_dict.keys())
    parser.add_argument('filename', help='checkpoint_filename')


    #parser.set_defaults(suffix="ditrl")
    #parser.set_defaults(generate_files=False)
    #parser.add_argument('--gen', help='generate_files', dest='generate_files', action='store_true')

    parser.add_argument('--frames', help='number of frames', default=64, type=int)
    #parser.set_defaults(swap_color=False)
    #parser.add_argument('--swap', help='switch black and white intensities', dest='swap_color', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_exec_args()
    lfd_params = default_model_params()
    lfd_params.set_application("tea_making")
    lfd_params.set_model_params(model_dict[args.model], end_point=-1)

    exec_func(args, lfd_params)
