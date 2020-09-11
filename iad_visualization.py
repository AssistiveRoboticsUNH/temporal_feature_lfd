"""
This code is for the training the pipeline section of D-ITR-L using the backbone model. This code also saves the frames
as ITRs
"""
import torch
import PIL
from PIL import Image, ImageDraw
import numpy as np
import os

from analysis.image_capture import *


def sparse_map_to_img(sparse_map, length):
    """
    Convert a sparse map to a binarized IAD
    """
    num_features = len(sparse_map)

    iad = np.zeros((num_features, length))
    print("iad shape:", iad.shape)
    for i, f in enumerate(sparse_map):
        print(i, f)
        print(iad[i].shape)
        for pair in f:
            iad[i, pair[0]:pair[1]] = 1
    iad *= -1
    iad += 1

    return iad


def run(lfd_params, model):
    # Create DataLoaders
    train_loader = lfd_params.create_dataloader(lfd_params, "train", shuffle=False, verbose=True)
    eval_loader = lfd_params.create_dataloader(lfd_params, "evaluation", shuffle=False, verbose=True)

    # put model on GPU
    rgb_model = model.rgb_net
    net = torch.nn.DataParallel(rgb_model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    # generate ITRs
    for data_loader in [train_loader, eval_loader]:
        for i, data_packet in enumerate(data_loader):

            obs, state, action, filename = data_packet

            # input shapes
            if i == 0:
                print("obs: ", obs.shape)
                print("state: ", state.shape)

            # obtain the Activation map
            activation_map = net(obs)
            print("am 1:", activation_map.shape)
            activation_map = activation_map.view((-1, lfd_params.args.num_segments) + activation_map.size()[1:])
            print("am 2:", activation_map.shape)
            activation_map = activation_map.detach().cpu().numpy()

            for n, file in enumerate(filename):

                # get IAD information (currently this is taken directly from the sparse map, as normalizing the IADs
                # is a challenge that will involve a lot of messy programming).
                print("am 3:", activation_map[n].shape)
                iad = model.pipeline.convert_activation_map_to_iad(activation_map[n])
                print("iad:", iad.shape)
                sparse_map = model.pipeline.convert_iad_to_sparse_map(iad)
                print("sm:", len(sparse_map))
                iad_img = sparse_map_to_img(sparse_map, lfd_params.args.num_segments)

                # get RGB frames
                rgb_image = read_file(lfd_params.args.num_segments, file, save_file=False, merge_images=False)

                new_frames = []
                for f, frame in enumerate(rgb_image):
                    # format iad_frame to work as an image
                    iad_frame = iad_img[:, f] * 255
                    iad_frame = np.uint8(iad_frame)
                    iad_frame = iad_frame.reshape(-1, 1)
                    iad_frame = Image.fromarray(iad_frame)

                    # resize the iad_frame
                    new_size = (512, frame.width)
                    iad_frame = iad_frame.resize(new_size, Image.NEAREST)

                    # add frame to list
                    frame = get_concat_v(frame, ImageDraw.rectangle([0, frame.width, 0, 10], fill=(255, 0, 0)))
                    new_frames.append(get_concat_v(frame, iad_frame))

                out_img = new_frames[0]
                for i in range(1, len(new_frames)):
                    out_img = get_concat_h(out_img, new_frames[i])

                # format new save name
                save_id = file.split('/')
                file_id = save_id[-1]+"_iad.png"
                save_id = "analysis/fig/"
                save_id = os.path.join(save_id, file_id)

                # save the image
                out_img.save(save_id)
                print("n: {0}, filename: {1}, save_id: {2}".format(n, file, save_id))

            print("generate ITRs: iter: {:6d}/{:6d}".format(i, len(data_loader)))

if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()

    from model.temporal_feature_extractor import TemporalFeatureExtractor
    model_obj = TemporalFeatureExtractor(lfd_params_obj, use_pipeline=True, train_pipeline=True, use_model=False,
                                         train_model=False)

    run(lfd_params_obj, model_obj)
