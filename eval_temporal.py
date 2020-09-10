"""
This code is for the training of a network using only the backbone model
"""
import torch
import numpy as np
import pandas as pd
import os


def evaluate(lfd_params, model, debug=True):

    # Create DataLoaders
    data_loader = lfd_params.create_dataloader(lfd_params, "evaluation", verbose=True)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    # recorded values
    rec_obs_label = []
    rec_state = []
    rec_expected_action = []
    rec_observed_action = []

    # Evaluate Network
    with torch.no_grad():

        for i, data_packet in enumerate(data_loader):

            obs, state, action, filename = data_packet

            # input shapes
            if debug and i == 0:
                print("obs_x: ", obs.shape)
                print("state_x: ", state.shape)

            # compute output
            action_logits = net(obs, state).detach().cpu().numpy()
            action_out = np.argmax(action_logits, axis=1)

            for j, file in enumerate(filename):
                # add information to DataFrame
                rec_obs_label.append(file.split('/')[-2])
                rec_state.append(state[j].detach().cpu().numpy())
                rec_expected_action.append(action[j].detach().cpu().numpy())
                rec_observed_action.append(action_out[j])

    # write output to file
    import datetime
    return pd.DataFrame({
        "obs_label": rec_obs_label,
        "state": rec_state,
        "expected_action": rec_expected_action,
        "observed_action": rec_observed_action,
    })


if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()
    lfd_params_obj.use_itrs()

    from model.temporal_feature_extractor import TemporalFeatureExtractor
    model_obj = TemporalFeatureExtractor(lfd_params_obj, use_pipeline=False, train_pipeline=False, use_model=True,
                                         train_model=False)

    df = evaluate(lfd_params_obj, model_obj)

    out_filename = os.path.join(lfd_params_obj.args.output_dir, "output_" + lfd_params_obj.args.save_id + ".csv")
    df.to_csv(out_filename)

    print("Output placed in: " + out_filename)

