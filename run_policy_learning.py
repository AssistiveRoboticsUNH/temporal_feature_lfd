import torch
import os
import numpy as np
import pandas as pd

from datasets.utils import create_dataloader

NUM_TOTAL_ACTIONS = 4
WIN_HIST = 3


def train(lfd_params, model, verbose=False, input_dtype="video", ablation=False):

    # Create DataLoaders
    assert input_dtype in ["video", "iad", "gcn"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if input_dtype == "video":
        from datasets.dataset_video_trace import DatasetVideoTrace as CustomDataset
    elif input_dtype == "iad":
        from datasets.dataset_iad_trace import DatasetIADTrace as CustomDataset
    else:
        from datasets.dataset_gcn_trace import DatasetGCNTrace as CustomDataset
    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "train", eval=False, trace_path=lfd_params.application.trace_file, verbose=True,
                            backbone=lfd_params.model.model_id, num_segments=lfd_params.input_frames, ablation=ablation)
    data_loader = create_dataloader(dataset, lfd_params, "train", shuffle=True)

    # put model on GPU
    params = list(model.parameters())
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.train()

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.Adam(params, lr=lfd_params.lr)

    # Train Network
    loss_record = []
    with torch.autograd.detect_anomaly():

        epoch = lfd_params.epochs
        for e in range(epoch):

            cumulative_loss = 0

            for i, data_packet in enumerate(data_loader):

                obs, act, obs_filename, act_filename = data_packet

                # constrain size to a history of 5 timesteps
                obs = obs[:, -WIN_HIST:]
                act = act[:, -WIN_HIST:]

                # obtain label
                label = act[:, -1]
                label = torch.argmax(label, dim=1)

                # hide label
                act[:, -1] = 0

                # compute output
                logits = net(obs.float(), act.float())

                # get loss
                loss = criterion(logits, label.long().cuda())
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()

                if verbose:
                    print("epoch: {:3d}/{:3d}".format(e, epoch))
                    print("loss:", loss.cpu().detach().numpy())
                    print("expected:", label.cpu().detach().numpy())
                    print("pred:", np.argmax(logits.cpu().detach().numpy(), axis=1))
                    print("logits:")
                    print(logits.cpu().detach().numpy())

                cumulative_loss += loss.cpu().detach().numpy()
            print("e:", e, "loss:", cumulative_loss)
            loss_record.append(cumulative_loss)

    # show loss over time, output placed in Log Directory
    import matplotlib.pyplot as plt
    plt.plot(loss_record)

    # add bells and whistles to plt
    plt.title(model.filename)
    plt.ylabel("loss")
    plt.tight_layout()

    # make sure log_dir exists
    log_dir = model.filename
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save plt to file
    fig_filename = os.path.join(log_dir, "train_loss.png")
    plt.savefig(fig_filename)

    # clear plt so I don't draw on top of my multiple images.
    plt.clf()

    return model

def evaluate_single_action(lfd_params, model, mode="evaluation", verbose=False, input_dtype="video"):

    # Create DataLoaders
    assert input_dtype in ["video", "iad", "gcn"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if input_dtype == "video":
        from datasets.dataset_video_trace import DatasetVideoTrace as CustomDataset
    elif input_dtype == "iad":
        from datasets.dataset_iad_trace import DatasetIADTrace as CustomDataset
    else:
        from datasets.dataset_gcn_trace import DatasetGCNTrace as CustomDataset
    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, mode, trace_path=lfd_params.application.trace_file, verbose=True,
                            backbone=model.backbone_id, num_segments=lfd_params.args.num_segments)
    data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    # Train Network
    expected_label_list = []
    predicted_label_list = []
    obs_filename_list = []
    trace_id = []

    with torch.no_grad():
        for i, data_packet in enumerate(data_loader):
            obs, act, obs_filenames, _ = data_packet

            for j in range(1, act.shape[1]):

                o = obs[:, :j]
                a = act[:, :j]

                # constrain size to a history of 5 timesteps
                o = o[:, -WIN_HIST:]
                a = a[:, -WIN_HIST:]

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

def evaluate_action_trace(lfd_params, model, mode="evaluation", verbose=False, input_dtype="video", ablation=False):
    # Create DataLoaders
    assert input_dtype in ["video", "iad", "gcn"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if input_dtype == "video":
        from datasets.dataset_video_trace import DatasetVideoTrace as CustomDataset
    elif input_dtype == "iad":
        from datasets.dataset_iad_trace import DatasetIADTrace as CustomDataset
    #elif input_dtype == "itr":
    #    from obsolete_files.dataset_itr_trace import DatasetITRTrace as CustomDataset
    else:
        from datasets.dataset_gcn_trace import DatasetGCNTrace as CustomDataset
    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, mode, eval=True, trace_path=lfd_params.application.trace_file,
                            verbose=True, backbone=lfd_params.model.model_id, num_segments=lfd_params.input_frames,
                            ablation=ablation)
    data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    # Train Network
    expected_label_list = []
    predicted_label_list = []
    obs_filename_list = []

    with torch.no_grad():
        for i, data_packet in enumerate(data_loader):
            obs, act, obs_filenames, _ = data_packet

            predicted_action_history = []

            for j in range(1, act.shape[1]+1):

                o = obs[:, :j]
                a = act[:, :j]

                # obtain label
                label = a[:, -1]
                label = torch.argmax(label, dim=1)

                # prepare a_history
                a_history = np.zeros((1, (len(predicted_action_history)+1), NUM_TOTAL_ACTIONS))
                for k in range(len(predicted_action_history)):
                    a_history[0, k, predicted_action_history[k]] = 1
                a_history = torch.from_numpy(a_history)

                o = o[:, -WIN_HIST:]
                a_history = a_history[:, -WIN_HIST:]

                # compute output
                logits = net(o.float(), a_history.float())

                # get label information
                expected_label = label.cpu().detach().numpy()
                predicted_label = np.argmax(logits.cpu().detach().numpy(), axis=1)

                predicted_action_history.append(predicted_label)

            # add data to lists to be returned
            act = act.cpu().detach().numpy()
            for j in range(act.shape[1]):
                if len(expected_label_list) <= j:
                    expected_label_list.append([])
                    predicted_label_list.append([])
                    obs_filename_list.append([])

                expected_label_list[j].append(np.argmax(act[0, j]))
                predicted_label_list[j].append(predicted_action_history[j][0])
                obs_filename_list[j].append(obs_filenames[j][0])

            if verbose:
                print("file: {:3d}/{:3d}".format(i, len(data_loader)))

                print("expected_label:", expected_label)
                print("predicted_label:", predicted_label)
                print("logits:")
                print(logits.cpu().detach().numpy())

    df_dict = {}
    for i in range(len(expected_label_list)):
        df_dict["expected_label_" + str(i)] = expected_label_list[i]
        df_dict["predicted_label_" + str(i)] = predicted_label_list[i]
        df_dict["obs_filename_" + str(i)] = obs_filename_list[i]

    # return Pandas dataframe
    return pd.DataFrame(df_dict)
