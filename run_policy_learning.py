import torch
import os
import numpy as np
import pandas as pd

from datasets.utils import create_dataloader
NUM_TOTAL_ACTIONS = 4


def train(lfd_params, model, verbose=False):

    # Create DataLoaders
    assert lfd_params.args.input_dtype in ["video", "itr"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if lfd_params.args.input_dtype == "video":
        from datasets.dataset_video_trace import DatasetVideoTrace as CustomDataset
    else:
        from datasets.dataset_itr_trace import DatasetITRTrace as CustomDataset
    dataset = CustomDataset(lfd_params.file_directory, "train", trace_path=lfd_params.args.trace_file, verbose=True,
                            num_segments=lfd_params.args.num_segments)
    data_loader = create_dataloader(dataset, lfd_params, "train", shuffle=True)

    # put model on GPU
    params = list(model.parameters())
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.train()

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # define optimizer
    if lfd_params.args.optimizer == "SGD":
        optimizer = torch.optim.SGD(params,
                                    lfd_params.args.lr,
                                    momentum=lfd_params.args.momentum,
                                    weight_decay=lfd_params.args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=lfd_params.args.lr)

    # Train Network
    loss_record = []
    with torch.autograd.detect_anomaly():

        epoch = lfd_params.args.epochs
        for e in range(epoch):

            cumulative_loss = 0

            for i, data_packet in enumerate(data_loader):
                print("")
                print("i: {:d}/{:d}".format(i, len(data_loader)))

                if i > 20:
                    break;

                obs, act = data_packet

                # obtain label
                label = act[:, -1]
                label = torch.argmax(label, dim=1)

                # hide label
                act[:, -1] = 0

                # compute output
                print("obs:", obs.shape, obs.dtype)
                print("act:", act.shape, act.dtype)
                logits = net(obs.float(), act.float())

                # get loss
                print("label:", label.shape, label.dtype)
                loss = criterion(logits, label.long().cuda())
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()

                if verbose and i % 100 == 0:
                    print("epoch: {:3d}/{:3d}".format(e, epoch))

                    print("loss:", loss.cpu().detach().numpy())
                    print("expected:", label.cpu().detach().numpy())
                    print("pred:", np.argmax(logits.cpu().detach().numpy(), axis=1))
                    print("logits:")
                    print(logits.cpu().detach().numpy())

                cumulative_loss += loss.cpu().detach().numpy()
            loss_record.append(cumulative_loss)

    # save trained model parameters
    model.save_model()

    # show loss over time, output placed in Log Directory
    import matplotlib.pyplot as plt
    plt.plot(loss_record)

    # add bells and whistles to plt
    plt.title(lfd_params.args.save_id)
    plt.ylabel("loss")
    plt.tight_layout()

    # make sure log_dir exists
    log_dir = os.path.join(lfd_params.args.log_dir, lfd_params.args.save_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save plt to file
    fig_filename = os.path.join(log_dir, "train_loss.png")
    plt.savefig(fig_filename)

    # clear plt so I don't draw on top of my multiple images.
    plt.clf()


def evaluate_single_action(lfd_params, model, mode="evaluation", verbose=False):

    # Create DataLoaders
    assert lfd_params.args.input_dtype in ["video", "itr"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if lfd_params.args.input_dtype == "video":
        from datasets.dataset_video_trace import DatasetVideoTrace as CustomDataset
    else:
        from datasets.dataset_itr_trace import DatasetITRTrace as CustomDataset
    dataset = CustomDataset(lfd_params.file_directory, mode, trace_path=lfd_params.args.trace_file, verbose=True)
    data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    # Train Network
    expected_label_list = []
    predicted_label_list = []
    obs_filename_list = []
    trace_id = []

    for i, data_packet in enumerate(data_loader):
        obs, act, obs_filenames, _ = data_packet

        for j in range(1, act.shape[1]):

            o = obs[:j]
            a = act[:j]

            # obtain label
            label = a[:, -1]
            label = torch.argmax(label, dim=1)

            # hide label
            a[:, -1] = 0

            # compute output
            logits = net(o.float(), a.float())

            # get label information
            expected_label = label.cpu().detach().numpy()
            predicted_label = np.argmax(logits.cpu().detach().numpy(), axis=1)

            # add data to lists to be returned
            expected_label_list.append(expected_label)
            predicted_label_list.append(predicted_label)
            obs_filename_list.append(obs_filenames[j-1])
            trace_id.append(i)

            if verbose:
                print("file: {:3d}/{:3d}".format(i, len(data_loader)))

                print("expected_label:", expected_label)
                print("predicted_label:", predicted_label)
                print("logits:")
                print(logits.cpu().detach().numpy())

    # return Pandas dataframe
    return pd.DataFrame({
        "expected_label": expected_label_list,
        "predicted_label": predicted_label_list,
        "obs_filename_list": obs_filename_list,
        "trace_id": trace_id,
    })


def evaluate_action_trace(lfd_params, model, mode="evaluation", verbose=False):
    # Create DataLoaders
    assert lfd_params.args.input_dtype in ["video", "itr"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if lfd_params.args.input_dtype == "video":
        from datasets.dataset_video_trace import DatasetVideoTrace as CustomDataset
    else:
        from datasets.dataset_itr_trace import DatasetITRTrace as CustomDataset
    dataset = CustomDataset(lfd_params.file_directory, mode, trace_path=lfd_params.args.trace_file, verbose=verbose)
    data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    # Train Network
    expected_label_list = []
    predicted_label_list = []
    obs_filename_list = []
    trace_id = []

    for i, data_packet in enumerate(data_loader):
        obs, act, obs_filenames, _ = data_packet

        predicted_action_history = []

        for j in range(1, act.shape[1]):

            o = obs[:j]
            a = act[:j]

            # obtain label
            label = a[:, -1]
            label = torch.argmax(label, dim=1)

            # prepare a_history
            a_history = np.zeros((len(predicted_action_history)+1), NUM_TOTAL_ACTIONS)
            for k in range(len(predicted_action_history)):
                a_history[k, predicted_action_history[k]] = 1

            # compute output
            logits = net(o.float(), a_history.float())

            # get label information
            expected_label = label.cpu().detach().numpy()
            predicted_label = np.argmax(logits.cpu().detach().numpy(), axis=1)

            predicted_action_history.append(predicted_label)

            # add data to lists to be returned
            expected_label_list.append(expected_label)
            predicted_label_list.append(predicted_label)
            obs_filename_list.append(obs_filenames[j - 1])
            trace_id.append(i)

            if verbose:
                print("file: {:3d}/{:3d}".format(i, len(data_loader)))

                print("expected_label:", expected_label)
                print("predicted_label:", predicted_label)
                print("logits:")
                print(logits.cpu().detach().numpy())

    # return Pandas dataframe
    return pd.DataFrame({
        "expected_label": expected_label_list,
        "predicted_label": predicted_label_list,
        "obs_filename_list": obs_filename_list,
        "trace_id": trace_id,
    })


if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()

    from model.spatial_feature_extractor import SpatialFeatureExtractor
    model_obj = SpatialFeatureExtractor(lfd_params_obj, is_training=True)

    train(lfd_params_obj, model_obj)

