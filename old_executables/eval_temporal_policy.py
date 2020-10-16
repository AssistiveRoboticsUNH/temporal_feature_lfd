"""
This code is for the training of a network using only the backbone model
"""
import torch
import pandas as pd
import numpy as np
import os


def eval_model(lfd_params, model, mode="evaluation"):

    # Create DataLoaders
    from datasets_old.block_construction_traces_dl import create_dataloader_itr
    eval_loader = create_dataloader_itr(lfd_params, mode, verbose=True)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    action_expected = []
    action_selected = []
    obs_file = []

    for i, data_packet in enumerate(eval_loader):

        # need to unpack obs_data and feed through in a batch
        obs_data, action_data, obs_src, act_src = data_packet

        print("obs_data: ", obs_data.shape, obs_data.dtype)
        print("action_data: ", action_data.shape, action_data.dtype)

        for j in range(1, action_data.shape[1]):

            # get data up until the current guess
            obs = obs_data[:, :j]
            act = action_data[:, :j]

            # prepare next action
            next_action = torch.unsqueeze(act[:, -1], 0)
            next_action = torch.argmax(next_action, dim=2)

            # compute output
            action_logits, _ = net(obs, act)
            action_selection = torch.argmax(action_logits, dim=1)

            # add output to dataframe lists
            action_expected.append(next_action[0].detach().cpu().numpy()[0])
            action_selected.append(action_selection.detach().cpu().numpy()[0])
            obs_file.append(obs_src)

    correct = [1 if action_expected[i] == action_selected[i] else 0 for i in range(len(action_expected))]

    print("action_expected:", action_expected)
    print("action_selected:", action_selected)
    print("correct:", correct)

    print("Accuracy: ", np.sum(correct)/float(len(action_expected)))

    return pd.DataFrame({
        "obs_file": obs_file,
        "action_expected": action_expected,
        "action_selected": action_selected,
    })


if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()
    lfd_params_obj.use_itrs()

    from model.model import LfDNetwork
    model_obj = LfDNetwork(lfd_params_obj, is_training=False)

    df = eval_model(lfd_params_obj, model_obj)

    out_filename = os.path.join(lfd_params_obj.args.output_dir, "output_policy" + lfd_params_obj.args.save_id + ".csv")
    df.to_csv(out_filename)