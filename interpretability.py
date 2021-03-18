#import cv2

# pass video through conv net.
# Find out where the features are prior to the max function
# scale and display.

import numpy as np
import os
import copy
from PIL import Image

from enums import suffix_dict, model_dict, Suffix
from parameter_parser import default_model_params

from model.classifier import Classifier
from execute import make_model_name
from datasets.dataset_video import DatasetVideo

import torch
from scipy.signal import savgol_filter


def save_png(iad, output_filename, swap_color=False):
    if swap_color:
        iad -= 1
        iad *= -1

    iad *= 255
    iad = iad.astype(np.uint8)
    iad = Image.fromarray(iad)

    iad.save(output_filename, "PNG")

def generate_iad_png(iad, min_values, max_values):
    iad -= min_values
    iad /= (max_values - min_values)

    iad = iad.T
    return iad


def exec_func(args, lfd_params):

    # define datasets
    train_files = DatasetVideo(lfd_params, lfd_params.application.file_directory, "train", verbose=True,
                               num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
    evaluation_files = DatasetVideo(lfd_params, lfd_params.application.file_directory, "evaluation", verbose=True,
                                    num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)

    # define model
    filename = make_model_name(args, lfd_params, backbone=False)
    model = Classifier(lfd_params, filename, model_dict[args.model], Suffix.NONE,
                       use_feature_extractor=True, train_feature_extractor=False,
                       use_bottleneck=True,
                       use_spatial=False, train_spatial=False,
                       use_pipeline=False, train_pipeline=False,
                       use_temporal=False, train_temporal=False)

    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    # generate feature presence
    for dataset_files in [train_files, evaluation_files]:
        for obs, label, filename in dataset_files:

            # compute output
            activation_map = net(obs)
            activation_map = activation_map.detach().cpu().numpy()
            print(activation_map.shape)
    """




















    # find values
    num_features = lfd_params.model.bottleneck_size

    '''
    for obs, label, filename in train_files:
        iad = obs.detach().cpu().numpy()
        iad = iad.T
    '''

    # generate images
    for dataset_files in [train_files, evaluation_files]:
        for obs, label, filename in dataset_files:
            iad = obs.detach().cpu().numpy()

            #'/home/mbc2004/datasets/BlockConstructionTimed/iad_vgg/evaluation/n/n_0.npz
            print("processing: "+filename)
            filename_split = filename.split('/')

            filename_id = filename_split[-1].split('.')[0]+".png"
            obs_id = filename_split[-2]
            mode_id = filename_split[-3]

            iad_png_dir = os.path.join(*[lfd_params.application.file_directory, "iad_png", mode_id, obs_id])
            event_png_dir = os.path.join(*[lfd_params.application.file_directory, "event_png", mode_id, obs_id])
            threshold_png_dir = os.path.join(*[lfd_params.application.file_directory, "threshold_png", mode_id, obs_id])

            if not os.path.exists(iad_png_dir):
                os.makedirs(iad_png_dir)
            if not os.path.exists(event_png_dir):
                os.makedirs(event_png_dir)
            if not os.path.exists(threshold_png_dir):
                    os.makedirs(threshold_png_dir)

            iad_output_filename = os.path.join(iad_png_dir, filename_id)
            scaled_iad = generate_iad_png(copy.deepcopy(iad), global_min_values, global_max_values)
            save_png(copy.deepcopy(scaled_iad), iad_output_filename, swap_color=args.swap_color)

            event_output_filename = os.path.join(event_png_dir, filename_id)
            event_iad = generate_event_png(copy.deepcopy(iad), global_avg_values)
            save_png(copy.deepcopy(event_iad), event_output_filename, swap_color=args.swap_color)

            threshold_output_filename = os.path.join(threshold_png_dir, filename_id)
            thresholded_iad = generate_threshold_png(copy.deepcopy(scaled_iad), copy.deepcopy(event_iad))
            save_png(thresholded_iad, threshold_output_filename, swap_color=args.swap_color)
    """


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize where features occur in the dataset')

    parser.add_argument('model', help='model_id', choices=model_dict.keys())

    parser.set_defaults(app='c')
    parser.set_defaults(suffix='backbone')

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
