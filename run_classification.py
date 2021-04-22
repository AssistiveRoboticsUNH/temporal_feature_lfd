import torch
import os
import numpy as np
import pandas as pd
from enums import Format

from datetime import datetime

from datasets.utils import create_dataloader


def train(lfd_params, model, verbose=False, input_dtype="video", overwrite_path=None):

    # Create DataLoaders
    assert input_dtype in ["video", "iad", "gcn"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if input_dtype == "video":
        from datasets.dataset_video import DatasetVideo as CustomDataset
        dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "train", verbose=False,
                                num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
        data_loader = create_dataloader(dataset, lfd_params, "train", shuffle=True)
    elif input_dtype == "iad":
        from datasets.dataset_iad import DatasetIAD as CustomDataset
        dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "train", verbose=False,
                                num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id,
                                overwrite_root_path=overwrite_path)
        data_loader = create_dataloader(dataset, lfd_params, "train", shuffle=True)
    else:
        from datasets.dataset_gcn import DatasetGCN as CustomDataset
        dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "train", verbose=False,
                                num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
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
    train_acc = []
    eval_acc = []
    start_time = datetime.now()

    with torch.autograd.detect_anomaly():

        epoch = lfd_params.epochs
        for e in range(epoch):

            cumulative_loss = 0

            for i, data_packet in enumerate(data_loader):

                obs, label = data_packet
                obs = obs.float()

                # compute output
                logits = net(obs)

                # get loss
                loss = criterion(logits, label.cuda())
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()
                if verbose and i % 1000 == 0:
                    print("epoch: {:3d}/{:3d}".format(e, epoch))
                    print("example: {:3d}/{:3d}".format(i, len(data_loader)))

                    print("loss:", loss.cpu().detach().numpy())
                    print("expected:", label.cpu().detach().numpy())
                    print("pred:", np.argmax(logits.cpu().detach().numpy(), axis=1))
                    #print("logits:")
                    #print(logits.cpu().detach().numpy())


                    now_time = datetime.now()
                    print("elapsed_time: ", str(now_time-start_time))

                cumulative_loss += loss.cpu().detach().numpy()
            print("e:", e, "loss:", cumulative_loss)
            loss_record.append(cumulative_loss)

    #model.save_model()

    # show loss over time, output placed in Log Directory
    import matplotlib.pyplot as plt
    plt.plot(loss_record)
    #plt.plot(train_acc)
    #plt.plot(eval_acc)

    # add bells and whistles to plt
    plt.title(model.filename)#lfd_params.args.save_id)
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


def evaluate(lfd_params, model, mode="evaluation", verbose=False, input_dtype="video", overwrite_path=None):

    # Create DataLoaders
    assert input_dtype in ["video", "iad", "gcn"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if input_dtype == "video":
        from datasets.dataset_video import DatasetVideo as CustomDataset
        dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "train", verbose=True,
                                num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
        data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=True)
    elif input_dtype == "iad":
        from datasets.dataset_iad import DatasetIAD as CustomDataset
        dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "train", verbose=True,
                                num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id,
                                overwrite_root_path=overwrite_path)
        data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=True)
    else:
        from datasets.dataset_gcn import DatasetGCN as CustomDataset
        dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "train", verbose=True,
                                num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
        data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=True)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    # Train Network
    expected_label_list = []
    predicted_label_list = []
    filename_list = []

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet
        obs = obs.float()

        # compute output
        logits = net(obs)

        # get label information
        expected_label = label.cpu().detach().numpy()[0]
        predicted_label = np.argmax(logits.cpu().detach().numpy(), axis=1) [0]

        # add data to lists to be returned
        expected_label_list.append(expected_label)
        predicted_label_list.append(predicted_label)
        filename_list.append(filename)

        if verbose and i % 1000 == 0:
            print("example: {:3d}/{:3d}".format(i, len(data_loader)))

            print("expected_label:", expected_label)
            print("predicted_label:", predicted_label)
            #print("logits:")
            #print(logits.cpu().detach().numpy())

    # return Pandas dataframe
    return pd.DataFrame({
        "expected_label": expected_label_list,
        "predicted_label": predicted_label_list,
        "filename": filename_list,
    })

def generate_iad_files(lfd_params, model, dataset_mode, verbose=False, backbone=None):

    # Create DataLoaders
    #assert lfd_params.input_dtype in ["video"], "ERROR: run_classification.py: input_dtype must be 'video'"

    if lfd_params.application.format == Format.VIDEO:
        from datasets.dataset_video import DatasetVideo as CustomDataset
        overwrite_root_path = None
        src_dir = "frames"
    elif lfd_params.application.format == Format.IAD:
        from datasets.dataset_iad import DatasetIAD as CustomDataset
        src_dir = "iad_src"
        overwrite_root_path = os.path.join(lfd_params.application.file_directory, src_dir)


    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, dataset_mode, verbose=True,
                            num_segments=lfd_params.input_frames, overwrite_root_path=overwrite_root_path)
    data_loader = create_dataloader(dataset, lfd_params, dataset_mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet

        # compute output
        iad = net(obs)
        iad = iad.detach().cpu().numpy()

        for n, file in enumerate(filename):

            # format new save name
            save_id = file.split('/')
            print("save_id:", save_id)

            file_id = save_id[-1]
            tail = save_id[save_id.index(src_dir) + 1:]
            print("tail:", tail)
            #save_id = save_id[:save_id.index(src_dir)] + ["iad_" + backbone] + tail
            if lfd_params.application.format == Format.VIDEO:
                file_id += ".npz"
            elif lfd_params.application.format == Format.IAD:
                tail = tail[:-1]
            save_dir = save_id[:save_id.index(src_dir)] + ["iad_" + backbone] + tail

            save_dir = '/' + os.path.join(*save_dir)

            # create a directory to save the ITRs in
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_id = os.path.join(save_dir, file_id)

            if verbose:
                print("n: {0}, filename: {1}, saved_id: {2}".format(n, file, save_id))
            # save ITR to file with given name
            print(save_id)
            print("iad.shape:", iad[n].shape)

            np.savez(save_id, data=iad[n])