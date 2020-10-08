"""
This code is for the training the pipeline section of D-ITR-L using the backbone model. This code also saves the frames
as ITRs
"""
import torch
import numpy as np
import os


def train_pipeline(lfd_params, model, debug=True):
    # Create DataLoaders
    train_loader = lfd_params.create_dataloader(lfd_params, "train", shuffle=False, verbose=True)
    eval_loader = lfd_params.create_dataloader(lfd_params, "evaluation", shuffle=False, verbose=True)

    # generate IAD mask and threshold values
    net = model.rgb_net
    net = torch.nn.DataParallel(net, device_ids=lfd_params.args.gpus).cuda()
    net.train()

    print("train_temporal_pipeline.py Bottleneck size:", lfd_params.args.bottleneck)

    from model.ditrl import DITRL_MaskFinder
    mask_and_threshold = DITRL_MaskFinder()
    for i, data_packet in enumerate(train_loader):

        obs, state, action, filename = data_packet

        # input shapes
        if debug and i == 0:
            print("obs: ", obs.shape)
            print("state: ", state.shape)

        # compute output
        activation_map = net(obs)
        activation_map = activation_map.view((-1, lfd_params.args.num_segments) + activation_map.size()[1:])
        activation_map = activation_map.detach().cpu().numpy()

        for iad in activation_map:
            mask_and_threshold.add_data(iad)

        print("find IAD mask and threshold: iter: {:6d}/{:6d}".format(i, len(train_loader)))

    mask, threshold = mask_and_threshold.gen_mask_and_threshold()
    # torch.cuda.empty_cache()

    # Prepare Pipeline post-processing steps
    model.pipeline.preprocessing = False
    model.pipeline.mask_idx = mask
    model.pipeline.threshold_values = threshold

    print("Mask allows for N features:", len(mask))
    print(mask)

    net = model
    net = torch.nn.DataParallel(net, device_ids=lfd_params.args.gpus).cuda()
    net.train()

    for i, data_packet in enumerate(train_loader):

        obs, state, action, filename = data_packet

        # input shapes
        if debug and i == 0:
            print("obs: ", obs.shape)
            print("state: ", state.shape)

        # compute output
        _ = net(obs)

        print("train pipeline: iter: {:6d}/{:6d}".format(i, len(train_loader)))

    model.fit_pipeline()

    # save trained model parameters
    out_filename = lfd_params.generate_modelname()
    model.pipeline.preprocessing = True
    model.pipeline.is_training = False
    model.save_model()

    # generate ITRs
    for data_loader in [train_loader, eval_loader]:
        for i, data_packet in enumerate(data_loader):

            obs, state, action, filename = data_packet

            # input shapes
            if i == 0:
                print("obs: ", obs.shape)
                print("state: ", state.shape)

            # compute output
            itrs = net(obs)
            itrs = itrs.detach().cpu().numpy()

            for n, file in enumerate(filename):

                # format new save name
                save_id = file.split('/')
                file_id = save_id[-1]+".npz"
                save_id = save_id[:save_id.index("frames")] + ["itrs"] + save_id[save_id.index("frames") + 1:-1]
                save_id = '/' + os.path.join(*save_id)

                # create a directory to save the ITRs in
                if not os.path.exists(save_id):
                    os.makedirs(save_id)

                save_id = os.path.join(save_id, file_id)

                if debug:
                    print("n: {0}, filename: {1}, saved_id: {2}".format(n, file, save_id))

                # save ITR to file with given name
                np.savez(save_id, data=itrs[n])

    return out_filename


if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()

    from model.temporal_feature_extractor import TemporalFeatureExtractor
    model_obj = TemporalFeatureExtractor(lfd_params_obj, use_pipeline=True, train_pipeline=True, use_model=False,
                                         train_model=False)

    train_pipeline(lfd_params_obj, model_obj)

