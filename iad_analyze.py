import numpy as np
import os
import copy
from PIL import Image

from enums import suffix_dict, model_dict, Suffix
from parameter_parser import default_model_params

from execute import generate_files
from datasets.dataset_iad import DatasetIAD

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


def generate_event_png(iad, avg_values):
    #for i in range(iad.shape[1]):
    #    iad[:, i] = savgol_filter(iad[:,i], 3, 1)
    iad = np.where(iad < avg_values, 0, 1)

    iad = iad.T
    return iad


def convert_iad_to_sparse_map(thresholded_iad):
    """Convert the IAD to a sparse map that denotes the start and stop times of each feature"""

    # apply threshold to get indexes where features are active
    locs = np.where(thresholded_iad)
    locs = np.dstack((locs[0], locs[1]))
    locs = locs[0]

    # get the start and stop times for each feature in the IAD
    if len(locs) != 0:
        sparse_map = []
        for i in range(thresholded_iad.shape[0]):
            feature_row = locs[np.where(locs[:, 0] == i)][:, 1]

            # locate the start and stop times for the row of features
            start_stop_times = []
            if len(feature_row) != 0:
                start = feature_row[0]
                for j in range(1, len(feature_row)):
                    if feature_row[j - 1] + 1 < feature_row[j]:
                        start_stop_times.append([start, feature_row[j - 1] + 1])
                        start = feature_row[j]

                start_stop_times.append([start, feature_row[len(feature_row) - 1] + 1])

            # add start and stop times to sparse_map
            sparse_map.append(start_stop_times)
    else:
        sparse_map = [[] for x in range(thresholded_iad.shape[0])]

    return sparse_map

def generate_threshold_png(scaled_iad, event_iad):

    #print("scaled_iad:", scaled_iad.shape)
    #print("event_iad:", event_iad.shape)
    #print("-----")

    sparse_map = convert_iad_to_sparse_map(event_iad)
    #print("len(sparse_map):", len(sparse_map))

    for f, feature in enumerate(sparse_map):
        #print("len(feature):", len(feature))
        temp = 0
        for (st, et) in feature:
            #print(f"st: {st} et: {et}")
            scaled_iad[f, temp:st] = 0
            scaled_iad[f, st:et] = np.max(scaled_iad[f, st:et])
            temp = et
        scaled_iad[f, temp:scaled_iad.shape[1]-1] = 0

    return scaled_iad



def exec_func(args, lfd_params):


    if args.generate_files:
        generate_files(args, lfd_params, backbone=False)

    train_files = DatasetIAD(lfd_params, lfd_params.application.file_directory, "train", verbose=True,
                             num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
    evaluation_files = DatasetIAD(lfd_params, lfd_params.application.file_directory, "evaluation", verbose=True,
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

    print("min:", global_min_values)
    print("max:", global_max_values)
    print("avg:", global_avg_values)

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
            save_png(scaled_iad, iad_output_filename)

            event_output_filename = os.path.join(event_png_dir, filename_id)
            event_iad = generate_event_png(copy.deepcopy(iad), global_avg_values)
            save_png(event_iad, event_output_filename)

            threshold_output_filename = os.path.join(threshold_png_dir, filename_id)
            thresholded_iad = generate_threshold_png(copy.deepcopy(scaled_iad), event_iad)
            save_png(thresholded_iad, threshold_output_filename)


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    parser.add_argument('model', help='model_id', choices=model_dict.keys())

    parser.set_defaults(suffix="ditrl")
    parser.set_defaults(generate_files=False)
    parser.add_argument('--gen', help='generate_files', dest='generate_files', action='store_true')

    parser.add_argument('--frames', help='number of frames', default=64, type=int)
    parser.set_defaults(swap_colors=False)
    parser.add_argument('--swap', help='switch black and white intensities', dest='swap_colors', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_exec_args()
    lfd_params = default_model_params()
    lfd_params.set_application("block_construction_timed")
    lfd_params.set_model_params(model_dict[args.model], end_point=-1)

    exec_func(args, lfd_params)
