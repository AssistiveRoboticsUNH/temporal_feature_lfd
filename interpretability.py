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


def convert_to_img(args, filename, activation_map, feature_ranking, max_features=5,
                   min_v_global=None, max_v_global=None, avg_v_global=None):

    #rgb_img = np.array([Image.open(f) for f in os.listdir(filename)])
    print("filename:", filename)
    total_num_frames = len(os.listdir(filename))
    idxs = np.linspace(0, max(1, total_num_frames - 1), num=args.frames, dtype=int) + 1
    rgb_img = [np.asarray(Image.open(os.path.join(filename, 'image_{:05d}.jpg'.format(idx))).convert('RGB').resize((224, 224))) for idx in idxs]
    rgb_img = np.array(rgb_img)


    print("rgb_img.shape:", rgb_img.shape)
    '''
    rgb_img = rgb_img.reshape([args.frames, 3, rgb_img.shape[-2], rgb_img.shape[-1]])
    rgb_img = rgb_img.transpose([0, 2, 3, 1])
    rgb_img *= 255
    rgb_img = rgb_img.astype(np.uint8)
    '''

    num_frames, height, width = rgb_img.shape[0], rgb_img.shape[1], rgb_img.shape[2]
    num_features = activation_map.shape[1]

    assert len(feature_ranking) == num_features, "ERROR: feature rankings don't match number of features"

    activation_map = activation_map.transpose([1, 0, 2, 3])
    print("activation_map:", activation_map.shape)



    for f in range(num_features):
        print(f"f: {f}, nf: {num_features}")

        if avg_v_global is not None:
            print(f"am_{f} pre:", activation_map[f], avg_v_global[f])
            activation_map[f][activation_map[f] < avg_v_global[f]] = -np.Inf
            print(f"am_{f} post:", activation_map[f])

        if min_v_global is None and max_v_global is None:
            min_v, max_v = np.max(activation_map[f]), np.min(activation_map[f])
        else:
            min_v, max_v = min_v_global[f], max_v_global[f]
        activation_map[f] = (activation_map[f] - min_v) / (max_v - min_v)

    activation_map[f][activation_map[f] > 1] = 1
    activation_map[f][activation_map[f] < 0] = 0

    #activation_map -= 1
    #activation_map *= -1
    activation_map *= 255
    activation_map = activation_map.astype(np.uint8)

    print("rgb_img.shape:", rgb_img.shape)
    print("activation_map.shape:", activation_map.shape)

    #dst = Image.new('RGB', (width * num_frames, height * num_features))
    dst = Image.new('RGB', (int(width/2) * num_frames, int(height/2) * max_features))
    #dst = Image.new('RGB', (width, height * num_features))
    #dst = Image.new('RGB', (width, height))

    for fi, f in enumerate(feature_ranking[:max_features]):
        for t in range(num_frames):

            img_frame = Image.fromarray(rgb_img[t]).convert("LA").convert("RGBA").resize((int(width/2), int(height/2)))

            activation_frame = Image.fromarray(activation_map[f, t])

            # create colored overlay
            activation_frame_dst = np.array(Image.new("HSV", activation_frame.size))
            hue = int((float(fi) / max_features) * 255)
            activation_frame_dst[..., 0] = hue
            activation_frame_dst[..., 1] = 100
            activation_frame_dst[..., 2] = 100

            # apply activation map as alpha channel
            activation_frame_dst = np.array(Image.fromarray(activation_frame_dst, 'HSV').convert("RGBA"))
            activation_frame_dst[..., 3] = activation_frame
            activation_frame_dst = Image.fromarray(activation_frame_dst, "RGBA").resize((int(width/2), int(height/2)), PIL.Image.NEAREST)

            # combine images
            img_frame = Image.alpha_composite(img_frame, activation_frame_dst)

            # add to full image
            dst.paste(img_frame, (int(width/2) * t, int(height/2) * fi))

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

    feature_ranking_file = pd.read_csv(os.path.join(args.filename, 'importance.csv'))

    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    # generate feature presence
    for dataset_files in [train_files, evaluation_files]:
        for obs, label, filename in dataset_files:

            if label in [0, 2, 3]:
                print("label: ", label)

                # get correct feature ranking
                feature_label = feature_ranking_file["feature"][feature_ranking_file["mode"] == "train"]
                feature_rank = feature_ranking_file["importance_label_"+str(label)][feature_ranking_file["mode"] == "train"]
                feature_ranking = list(zip(feature_rank, feature_label))
                feature_ranking.sort()
                _, feature_ranking = zip(*feature_ranking)
                print("feature_ranking:", feature_ranking)

                # compute output
                activation_map = net(obs)
                obs = obs.detach().cpu().numpy()
                activation_map = activation_map.detach().cpu().numpy()
                print(activation_map.shape)

                img_out = convert_to_img(args, filename, activation_map, feature_ranking=feature_ranking, max_features=args.max)

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

def exec_func_global(args, lfd_params):
    from datasets.dataset_iad import DatasetIAD

    train_files = DatasetIAD(lfd_params, lfd_params.application.file_directory, "train", verbose=True,
                             num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)

    # find values
    num_features = lfd_params.model.bottleneck_size

    global_min_values = np.zeros(num_features)
    global_max_values = np.zeros(num_features)
    global_avg_values = np.zeros(num_features)
    global_cnt_values = 0

    for obs, label, filename in train_files:
        iad = obs.detach().cpu().numpy()
        iad = iad.T

        min_values = np.min(iad, axis=1)
        max_values = np.max(iad, axis=1)
        avg_values = np.sum(iad, axis=1)
        cnt_values = iad.shape[1]

        # update globals
        for i, v in enumerate(min_values):
            if v < global_min_values[i]:
                global_min_values[i] = v

        for i, v in enumerate(max_values):
            if v > global_max_values[i]:
                global_max_values[i] = v

        global_avg_values *= global_cnt_values
        global_cnt_values += cnt_values
        global_avg_values += avg_values
        global_avg_values /= global_cnt_values

    print("aggregating data")



    # GENERATE IMG

    # define datasets
    train_files = DatasetVideo(lfd_params, lfd_params.application.file_directory, "train", verbose=True,
                               num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id,
                               specific_labels=[0, 2, 3])
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

    feature_ranking_file = pd.read_csv(os.path.join(args.filename, 'importance.csv'))

    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    # generate feature presence
    counter = 0
    for dataset_files in [train_files]:#, evaluation_files]:
        for obs, label, filename in dataset_files:

            print("counter:", counter)
            counter+=1

            if obs is not None:


                print("label: ", label)

                # get correct feature ranking
                feature_label = feature_ranking_file["feature"][feature_ranking_file["mode"] == "train"]
                feature_rank = feature_ranking_file["importance_label_"+str(label)][feature_ranking_file["mode"] == "train"]
                feature_ranking = list(zip(feature_rank, feature_label))
                feature_ranking.sort()
                _, feature_ranking = zip(*feature_ranking)
                print("feature_ranking:", feature_ranking)

                # compute output
                activation_map = net(obs)
                obs = obs.detach().cpu().numpy()
                activation_map = activation_map.detach().cpu().numpy()
                print(activation_map.shape)

                img_out = convert_to_img(args, filename, activation_map, feature_ranking=feature_ranking,
                                         max_features=args.max,
                                         min_v_global=min_values, max_v_global=max_values, avg_v_global=global_avg_values)

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
    parser.add_argument('--max', help='max features to show', default=5, type=int)
    parser.add_argument('--application', help='application', default="block_construction_timed",
                        choices=['block_construction_timed', 'block_construction', 'tea_making'])
    #parser.set_defaults(swap_color=False)
    #parser.add_argument('--swap', help='switch black and white intensities', dest='swap_color', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_exec_args()
    lfd_params = default_model_params()
    lfd_params.set_application(args.application)
    lfd_params.set_model_params(model_dict[args.model], end_point=-1)

    exec_func_global(args, lfd_params)
