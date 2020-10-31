import torch
import os
import numpy as np
import pandas as pd

from datasets.utils import create_dataloader


def train(lfd_params, model, verbose=False, input_dtype="video"):

    # Create DataLoaders
    assert input_dtype in ["video", "itr", "gcn"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if input_dtype == "video":
        from datasets.dataset_video_trace import DatasetVideoTrace as CustomDataset
    elif input_dtype == "gcn":
        from datasets.dataset_itr_trace import DatasetITRTrace as CustomDataset
    else:
        from datasets.dataset_itr_trace import DatasetGCNTrace as CustomDataset
    dataset = CustomDataset(lfd_params.file_directory, "train", verbose=False,
                            num_segments=lfd_params.args.num_segments, backbone=model.backbone_id)
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
            loss_record.append(cumulative_loss)

    # show loss over time, output placed in Log Directory
    import matplotlib.pyplot as plt
    plt.plot(loss_record)

    # add bells and whistles to plt
    plt.title(lfd_params.args.save_id)
    plt.ylabel("loss")
    plt.tight_layout()

    # make sure log_dir exists
    log_dir = lfd_params.args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save plt to file
    fig_filename = os.path.join(log_dir, lfd_params.args.save_id+"_train_loss.png")
    plt.savefig(fig_filename)

    # clear plt so I don't draw on top of my multiple images.
    plt.clf()

    return model


def evaluate(lfd_params, model, mode="evaluation", verbose=False, input_dtype="video"):

    # Create DataLoaders
    assert input_dtype in ["video", "itr", "gcn"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if input_dtype == "video":
        from datasets.dataset_video_trace import DatasetVideoTrace as CustomDataset
    elif input_dtype == "gcn":
        from datasets.dataset_itr_trace import DatasetITRTrace as CustomDataset
    else:
        from datasets.dataset_itr_trace import DatasetGCNTrace as CustomDataset
    dataset = CustomDataset(lfd_params.file_directory, mode, verbose=True,
                            num_segments=lfd_params.args.num_segments, backbone=model.backbone_id)
    data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
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


if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()

    from model.spatial_feature_extractor import SpatialFeatureExtractor
    model_obj = SpatialFeatureExtractor(lfd_params_obj, is_training=True)

    train(lfd_params_obj, model_obj)

