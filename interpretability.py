#import cv2

# pass video through conv net.
# Find out where the features are prior to the max function
# scale and display.

import numpy as np
import os
import copy
import PIL
from PIL import Image
import pandas as pd

from enums import suffix_dict, model_dict, Suffix
from parameter_parser import default_model_params

from model.classifier import Classifier
from execute import make_model_name
from datasets.dataset_video import DatasetVideo

import torch
from scipy.signal import savgol_filter


def convert_to_img(args, rgb_img, activation_map, feature_ranking):

    print("rgb_img.shape:", rgb_img.shape)
    rgb_img = rgb_img.reshape([args.frames, 3, rgb_img.shape[-2], rgb_img.shape[-1]])
    rgb_img = rgb_img.transpose([0, 2, 3, 1])
    rgb_img *= 255
    rgb_img = rgb_img.astype(np.uint8)

    num_frames, height, width = rgb_img.shape[0], rgb_img.shape[1], rgb_img.shape[2]
    num_features = activation_map.shape[1]

    assert len(feature_ranking) == num_features, "ERROR: feature rankings don't match number of features"

    activation_map = activation_map.transpose([1, 0, 2, 3])
    print("activation_map:", activation_map.shape)
    for f in range(num_features):
        print(f"f: {f}, nf: {num_features}")
        min_v, max_v = np.max(activation_map[f]), np.min(activation_map[f])
        activation_map[f] = (activation_map[f] - min_v) / (max_v - min_v)
    activation_map -= 1
    activation_map *= -1
    activation_map *= 255
    activation_map = activation_map.astype(np.uint8)

    print("rgb_img.shape:", rgb_img.shape)
    print("activation_map.shape:", activation_map.shape)

    dst = Image.new('RGB', (width * num_frames, height * num_features))
    #dst = Image.new('RGB', (width, height * num_features))
    #dst = Image.new('RGB', (width, height))

    for f in feature_ranking:
        for t in range(num_frames):

            img_frame = Image.fromarray(rgb_img[t]).convert("LA").convert("RGBA")

            activation_frame = Image.fromarray(activation_map[f, t])

            # create colored overlay
            activation_frame_dst = np.array(Image.new("HSV", activation_frame.size))
            hue = int((float(f) / num_features) * 255)
            activation_frame_dst[..., 0] = hue
            activation_frame_dst[..., 1] = 100
            activation_frame_dst[..., 2] = 100

            # apply activation map as alpha channel
            activation_frame_dst = np.array(Image.fromarray(activation_frame_dst, 'HSV').convert("RGBA"))
            activation_frame_dst[..., 3] = activation_frame
            activation_frame_dst = Image.fromarray(activation_frame_dst, "RGBA").resize((width, height), PIL.Image.NEAREST)

            # combine images
            img_frame = Image.alpha_composite(img_frame, activation_frame_dst)

            # add to full image
            dst.paste(img_frame, (width * t, height * f))

    return dst


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

    feature_ranking = pd.read_csv(os.path.join(args.filename, 'importance.csv'))
    print("feature_ranking:")
    for f, r in enumerate(feature_ranking):
        print(f"f: {f}, r: {r}")

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

            img_out = convert_to_img(args, obs, activation_map, feature_ranking=feature_ranking)

            filename_split = filename.split('/')
            filename_id = filename_split[-1].split('.')[0] + ".png"
            obs_id = filename_split[-2]
            mode_id = filename_split[-3]
            png_dir = os.path.join(*[lfd_params.application.file_directory, "intr_png", mode_id, obs_id])
            if not os.path.exists(png_dir):
                os.makedirs(png_dir)
            output_filename = os.path.join(png_dir, filename_id)

            print("output_filename:", output_filename)
            img_out.save(output_filename, "PNG")
            print("done")


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
