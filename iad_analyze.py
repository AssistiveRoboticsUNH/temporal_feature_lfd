import numpy as np
import os
import copy
from PIL import Image

from enums import suffix_dict, model_dict, Suffix
from parameter_parser import default_model_params

from execute import generate_files
from datasets.dataset_iad import DatasetIAD


def generate_iad_png(iad, min_values, max_values, output_filename):
    iad -= min_values
    iad /= (max_values - min_values)

    iad = iad.T

    iad *= 255
    iad = iad.astype(np.uint8)
    iad = Image.fromarray(iad)

    iad.save(output_filename, "PNG")
    #print(output_filename)


def generate_event_png(iad, avg_values, output_filename):
    #print(iad)
    #print("iad1:", iad[:5])
    #print("thresh:", avg_values[:5])


    #print("avg_values:", avg_values)
    #print("iad:", iad)

    #print("iad.shape:", iad.shape)
    #print("avg_values:", avg_values.shape)

    print("avg: ", avg_values)
    mx = np.max(iad, axis=0)
    print("max: ", mx)
    mn = np.min(iad, axis=0)
    print("min: ", mn)

    print("avg.shape:", avg_values.shape)
    print("mx.shape:", mx.shape)

    print("iad1:")
    print(iad)
    iad[iad < avg_values] = 0
    #iad[iad >= avg_values] = 1
    #iad = np.where(iad < avg_values, np.zeros_like(iad), np.ones_like(iad))

    #print("iad.shape2:", iad.shape)
    print("iad2:")
    print(iad)

    iad = iad.T

    iad *= 255
    iad = iad.astype(np.uint8)
    iad = Image.fromarray(iad)

    iad.save(output_filename, "PNG")
    #print(output_filename)



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

        #print("iad.shape:", iad.shape)

        min_values = np.min(iad, axis=1)
        max_values = np.max(iad, axis=1)
        avg_values = np.sum(iad, axis=1)
        cnt_values = iad.shape[0]

        print("min_values:", min_values.shape)
        print("max_values:", max_values.shape)
        print("avg_values:", avg_values.shape)
        print("cnt_values:", cnt_values)

        # update globals
        for i, v in enumerate(min_values):
            if v < global_min_values[i]:
                global_min_values[i] = v

        for i, v in enumerate(max_values):
            if v > global_max_values[i]:
                global_max_values[i] = v

        print("avg1:", global_avg_values[:5])
        global_avg_values *= global_cnt_values
        print("avg2:", global_avg_values[:5])
        global_cnt_values += cnt_values
        print("new_vals:", avg_values[:5])
        global_avg_values += avg_values
        print("avg3:", global_avg_values[:5])
        global_avg_values /= global_cnt_values
        print("avg4:", global_avg_values[:5])


    print("min:", global_min_values)
    print("max:", global_max_values)
    print("avg:", global_avg_values)
    ''' 
    # generate images
    for dataset_files in [train_files, evaluation_files]:
        for obs, label, filename in dataset_files:
            iad = obs.detach().cpu().numpy()

            #for x, i in enumerate(global_max_values):
            #    if x in iad[i]:
            #        print("has max value:", x)
            #iad = iad.T

            #'/home/mbc2004/datasets/BlockConstructionTimed/iad_vgg/evaluation/n/n_0.npz
            print("processing: "+filename)
            filename_split = filename.split('/')

            filename_id = filename_split[-1].split('.')[0]+".png"
            obs_id = filename_split[-2]
            mode_id = filename_split[-3]

            iad_png_dir = os.path.join(*[lfd_params.application.file_directory, "iad_png",
                                         mode_id, obs_id])
            event_png_dir = os.path.join(*[lfd_params.application.file_directory, "event_png",
                                           mode_id, obs_id])

            if not os.path.exists(iad_png_dir):
                os.makedirs(iad_png_dir)
            if not os.path.exists(event_png_dir):
                os.makedirs(event_png_dir)

            iad_output_filename = os.path.join(iad_png_dir, filename_id)
            generate_iad_png(copy.deepcopy(iad), global_min_values, global_max_values, iad_output_filename)

            event_output_filename = os.path.join(event_png_dir, filename_id)
            generate_event_png(copy.deepcopy(iad), global_avg_values, event_output_filename)
    '''

def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    parser.add_argument('model', help='model_id', choices=model_dict.keys())

    parser.set_defaults(suffix="ditrl")
    parser.set_defaults(generate_files=False)
    parser.add_argument('--gen', help='generate_files', dest='generate_files', action='store_true')

    parser.add_argument('--frames', help='number of frames', default=64, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_exec_args()
    lfd_params = default_model_params()
    lfd_params.set_application("block_construction_timed")
    lfd_params.set_model_params(model_dict[args.model], end_point=-1)

    exec_func(args, lfd_params)
