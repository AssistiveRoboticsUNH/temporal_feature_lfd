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

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.train()

    # Train Network
    for i, data_packet in enumerate(train_loader):

        obs, state, action, filename = data_packet

        # input shapes
        if debug and i == 0:
            print("obs: ", obs.shape)
            print("state: ", state.shape)

        # compute output
        _ = net(obs)

        print("train pipeline: iter: {:6d}/{:6d}".format(i, len(train_loader)))

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

    # save trained model parameters
    out_filename = lfd_params.generate_modelname()
    model.save_model()

    return out_filename


if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()

    from model.temporal_feature_extractor import TemporalFeatureExtractor
    model_obj = TemporalFeatureExtractor(lfd_params_obj, use_pipeline=True, train_pipeline=True, use_model=False,
                                         train_model=False)

    train_pipeline(lfd_params_obj, model_obj)

