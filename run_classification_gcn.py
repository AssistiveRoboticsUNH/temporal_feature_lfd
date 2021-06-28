import torch
import os
import numpy as np
import pandas as pd

from datasets.utils_gcn import create_dataloader


def train(lfd_params, model, verbose=False, input_dtype="video"):

    # Create DataLoaders
    assert input_dtype in ["video", "gcn"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if input_dtype == "video":
        from datasets.dataset_video import DatasetVideo as CustomDataset
    #elif input_dtype == "itr":
    #    from obsolete_files.dataset_itr import DatasetITR as CustomDataset
    else:
        from datasets.dataset_gcn import DatasetGCN as CustomDataset

    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "train", verbose=False,
                            num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
    data_loader = create_dataloader(dataset, lfd_params, "train", shuffle=True)
    #data_loader = create_dataloader(dataset, lfd_params, "train", shuffle=True, batch_size=1)

    # put model on GPU
    print("RUNNING ON GPU: ", lfd_params.gpus)
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
    with torch.autograd.detect_anomaly():

        epoch = lfd_params.epochs
        for e in range(epoch):

            cumulative_loss = 0

            for i, data_packet in enumerate(data_loader):
                obs, label = data_packet

                # compute output
                logits = net(obs)

                # get loss
                loss = criterion(logits, label.cuda())
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
            print("e:", e, "loss:", cumulative_loss)
            loss_record.append(cumulative_loss)

    # show loss over time, output placed in Log Directory
    import matplotlib.pyplot as plt
    plt.plot(loss_record)
    #plt.plot(train_acc)
    #plt.plot(eval_acc)

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


def evaluate(lfd_params, model, mode="evaluation", verbose=False, input_dtype="video"):

    # Create DataLoaders
    assert input_dtype in ["video", "gcn"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if input_dtype == "video":
        from datasets.dataset_video import DatasetVideo as CustomDataset
    #elif input_dtype == "itr":
    #    from obsolete_files.dataset_itr import DatasetITR as CustomDataset
    else:
        from datasets.dataset_gcn import DatasetGCN as CustomDataset
    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, mode, verbose=True,
                            num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
    data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    # Train Network
    expected_label_list = []
    predicted_label_list = []
    filename_list = []

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet

        # compute output
        logits = net(obs)

        # get label information
        expected_label = label.cpu().detach().numpy()[0]
        predicted_label = np.argmax(logits.cpu().detach().numpy(), axis=1) [0]

        # add data to lists to be returned
        expected_label_list.append(expected_label)
        predicted_label_list.append(predicted_label)
        filename_list.append(filename)

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
        "filename": filename_list,
    })
