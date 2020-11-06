from datasets_old.video_dataset import VideoDataset
from datasets.dataset_video import DatasetVideo

import os
from PIL import Image, ImageFilter
import numpy as np 

import argparse


# need to remove ros path before I can import cv2
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def sparse_map_to_img(sparse_map, length):
    """
    Convert a sparse map to a binarized IAD
    """
    num_features = len(sparse_map)

    iad = np.zeros((num_features, length))
    for i, f in enumerate(sparse_map):
        for pair in f:
            iad[i, pair[0]:pair[1]] = 1
    #iad *= -1
    #iad += 1

    return iad

if __name__ == '__main__':

    # Usage: just run the file on my shield_ip
    # python3 view_dataset.py

    num_segments = 16
    image_tmpl = "image_{:05d}.jpg"
    full_sample = False

    img_dict = {}

    root_path = os.path.join("/home/mbc2004/", "datasets/BlockConstructionV2/")
    print("root_path:", root_path)

    save_id = "policy_learning_ditrl_tsm_bn16_2"
    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    filename = os.path.join(dir_name, "model")



    model_p = "tsm"

    from model_def import define_model
    model_dict = define_model(model_p)
    Classifier = model_dict["classifier"]
    PolicyLearner = model_dict["policy_learner"]
    num_segments = model_dict["num_segments"]
    bottleneck_size = model_dict["bottleneck_size"]
    dense_sample = model_dict["dense_sample"]
    dense_rate = model_dict["dense_rate"]

    from parameter_parser import parse_model_args, default_model_args
    lfd_params = default_model_args(save_id=save_id, log_dir="",
                                    num_segments=num_segments, bottleneck_size=bottleneck_size,
                                    dense_sample=False, dense_rate=8)

    model1 = PolicyLearner(lfd_params, filename, use_feature_extractor=True, use_spatial=False, use_pipeline=False,
                          use_temporal=False,
                          spatial_train=False, ditrl_pipeline_train=False, temporal_train=False)
    model2 = PolicyLearner(lfd_params, filename, use_feature_extractor=False, use_spatial=False, use_pipeline=True,
                          use_temporal=False,
                          spatial_train=False, ditrl_pipeline_train=False, temporal_train=False)
    import torch
    feature_extractor_net = torch.nn.DataParallel(model1, device_ids=lfd_params.args.gpus).cuda()
    #pipeline_net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()

    # generate images from entire dataset
    vd = DatasetVideo(lfd_params, root_path, "train", image_tmpl=image_tmpl, num_segments=num_segments)
    for i in range(len(vd)):
        print("i:", i, len(vd))
        data, label = vd[i]
        img = vd.show(i)


        # get un-binarized IAD
        am = feature_extractor_net(data, np.zeros(1)).detach().cpu().numpy()
        print("am shape:", am.shape)
        iad = model2.pipeline.pipeline.convert_activation_map_to_iad(am[0])
        print("iad:", iad.shape)
        min_v = iad.min(axis=0)
        print("min_v:", min_v.shape)
        max_v = iad.max(axis=0)
        iad_img = (iad - min_v) / (max_v - min_v)
        print("iad_img:", iad_img)
        print("thresh:", model2.pipeline.pipeline.threshold)

        # get binarized IAD
        print("iad:", iad.shape)
        sparse_map = model2.pipeline.pipeline.convert_iad_to_sparse_map(iad)
        print("sparse_map:", len(sparse_map))
        bin_iad_img = sparse_map_to_img(sparse_map, num_segments)
        print("bin_iad_img:", iad_img.shape)


        new_frames = []
        frame_h = img.height
        frame_w = int(img.width / num_segments)

        print("frame_h:", frame_h, "frame_w:", frame_w)
        for f in range(num_segments):
            # format iad_frame to work as an image
            iad_frame = iad_img[:, f] * 255
            iad_frame = np.uint8(iad_frame)
            iad_frame = iad_frame.reshape(-1, 1)
            iad_frame = Image.fromarray(iad_frame)

            # resize the iad_frame
            new_size = (frame_h, frame_w)
            iad_frame = iad_frame.resize(new_size, Image.NEAREST)

            # format iad_frame to work as an image
            bin_iad_frame = bin_iad_img[:, f] * 255
            bin_iad_frame = np.uint8(bin_iad_frame)
            bin_iad_frame = bin_iad_frame.reshape(-1, 1)
            bin_iad_frame = Image.fromarray(bin_iad_frame)

            # resize the iad_frame
            new_size = (frame_h, frame_w)
            bin_iad_frame = bin_iad_frame.resize(new_size, Image.NEAREST)

            # create image frame
            buffer_height = frame_h + 10
            iad_height = frame_h
            total_height = buffer_height*2 + iad_height

            large_frame = Image.new('RGB', (frame_w, total_height), color=(255, 0, 0))

            # add frame to list
            large_frame.paste(img.crop((frame_w * f, 0, frame_w * (f + 1), frame_h)), (0, 0))
            large_frame.paste(iad_frame, (0, buffer_height))
            large_frame.paste(bin_iad_frame, (0, buffer_height*2))

            new_frames.append(large_frame)  # get_concat_v(frame, iad_frame))

        out_img = new_frames[0]
        for z in range(1, len(new_frames)):
            out_img = get_concat_h(out_img, new_frames[z])


        out_img.save("analysis/dataset_fig/"+str(i).zfill(2)+"_clean.png")
        print("analysis/dataset_fig/"+str(i).zfill(2)+"_clean.png")



