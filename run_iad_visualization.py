import torch
import os
import numpy as np
import pandas as pd
from PIL import Image

from parameter_parser import parse_model_args, default_model_args
from model.classifier_ditrl_tsm import ClassifierDITRLTSM


def sparse_map_to_img(sparse_map, length):
    """
    Convert a sparse map to a binarized IAD
    """
    num_features = len(sparse_map)

    iad = np.zeros((num_features, length))
    for i, f in enumerate(sparse_map):
        for pair in f:
            iad[i, pair[0]:pair[1]] = 1
    return iad


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def visualize(lfd_params, model, mode="evaluation"):

    # Create DataLoaders
    from datasets.dataset_video import DatasetVideo
    dataset = DatasetVideo(lfd_params.file_directory, mode, verbose=True,
                           num_segments=lfd_params.args.num_segments, backbone=model.backbone_id)

    # put model on GPU
    net = torch.nn.DataParallel(model.backbone, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    for i, data_packet in enumerate(dataset):

        obs, label, filename = data_packet
        rgb_image = dataset.show(i)

        backbone_out = net(obs)  # pass in image, dont use pipeline, do use bottleneck
        backbone_out = backbone_out.detach().cpu().numpy()
        print("activation_map:", backbone_out.shape)
        #assert False

        # get IAD information (currently this is taken directly from the sparse map, as normalizing the IADs
        # is a challenge that will involve a lot of messy programming).
        iad = model.pipeline.pipeline.convert_activation_map_to_iad(backbone_out)
        sparse_map = model.pipeline.pipeline.convert_iad_to_sparse_map(iad)

        iad_img = sparse_map_to_img(sparse_map, lfd_params.args.num_segments)

        # view IAD frames
        new_frames = []
        frame_h = rgb_image.height
        frame_w = int(rgb_image.width / lfd_params.args.num_segments)

        print("frame_h:", frame_h, "frame_w:", frame_w)
        for f in range(lfd_params.args.num_segments):

            # format iad_frame to work as an image
            iad_frame = iad_img[:, f] * 255
            iad_frame = np.uint8(iad_frame)
            iad_frame = iad_frame.reshape(-1, 1)
            iad_frame = Image.fromarray(iad_frame)

            # resize the iad_frame
            new_size = (frame_h, frame_w)
            iad_frame = iad_frame.resize(new_size, Image.NEAREST)

            # create image frame
            buffer_height = frame_h + 10
            iad_height = frame_h
            total_height = buffer_height + iad_height

            large_frame = Image.new('RGB', (frame_w, total_height), color=(255, 0, 0))

            # add frame to list
            large_frame.paste(rgb_image.crop((frame_w * f, 0, frame_w * (f + 1), frame_h)), (0, 0))
            large_frame.paste(iad_frame, (0, buffer_height))

            new_frames.append(large_frame)

        out_img = new_frames[0]
        for z in range(1, len(new_frames)):
            out_img = get_concat_h(out_img, new_frames[z])

        # save the image
        print(filename)
        out_id = filename.split('/')[-1].split('.')[0]
        out_img.save("analysis/fig/viz_"+out_id+".png")


if __name__ == '__main__':

    save_id = "policy_learning_ditrl_tsm_bn16_2"
    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    assert os.path.exists(dir_name), "ERROR: run_iad_visualization.py: model filename not found"
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name, num_segments=32)

    model = ClassifierDITRLTSM(lfd_params, filename, use_feature_extractor=True, use_spatial=False,
                               use_pipeline=True, use_temporal=False,
                               spatial_train=False, ditrl_pipeline_train=False, temporal_train=False)

    visualize(lfd_params, model)

