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

    model = PolicyLearner(lfd_params, filename, use_feature_extractor=True, use_spatial=False, use_pipeline=False,
                          use_temporal=False,
                          spatial_train=False, ditrl_pipeline_train=False, temporal_train=False)
    import torch
    feature_extractor_net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    #pipeline_net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()

    # generate images from entire dataset
    vd = DatasetVideo(lfd_params, root_path, "train", image_tmpl=image_tmpl, num_segments=num_segments)
    for i in range(len(vd)):
        print("i:", i, len(vd))
        data, label = vd[i]

        am = feature_extractor_net(data)
        print("am shape:", am.shape)

        img = vd.show(i)
        img.save("analysis/dataset_fig/"+str(i).zfill(2)+"_clean.png")



