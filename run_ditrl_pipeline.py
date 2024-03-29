import torch
import os
import numpy as np

from datasets.utils import create_dataloader
from model.temporal.ditrl_gcn import DITRL_MaskFinder

from datasets.dataset_iad import DatasetIAD as CustomDataset


def train_pipeline(lfd_params, model):

    # Create DataLoaders
    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "train", num_segments=lfd_params.input_frames)
    data_loader = create_dataloader(dataset, lfd_params, "train", shuffle=False)

    # record mask and threshold values
    mask_and_threshold = DITRL_MaskFinder()

    for i, data_packet in enumerate(data_loader):
        activation_map, label = data_packet
        activation_map = activation_map.detach().cpu().numpy()

        for iad in activation_map:
            mask_and_threshold.add_data(iad)

    mask, threshold = mask_and_threshold.gen_mask_and_threshold()
    model.pipeline.pipeline.preprocessing = False
    model.pipeline.pipeline.mask_idx = mask
    model.pipeline.pipeline.threshold_values = threshold

    return model


def generate_binarized_iad_files(lfd_params, model, dataset_mode, verbose=False, backbone="tsm"):

    # Create DataLoaders
    assert lfd_params.args.input_dtype in ["video", "itr"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if lfd_params.args.input_dtype == "video":
        from datasets.dataset_video import DatasetVideo as CustomDataset

    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, dataset_mode, verbose=True,
                            num_segments=lfd_params.args.num_segments)
    data_loader = create_dataloader(dataset, lfd_params, dataset_mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet

        # compute output
        iads = net(obs)
        iads = iads.detach().cpu().numpy()

        for n, file in enumerate(filename):

            # format new save name
            save_id = file.split('/')
            file_id = save_id[-1] + ".npz"
            save_id = save_id[:save_id.index("frames")] + ["iad_"+backbone] + save_id[save_id.index("frames") + 1:-1]
            save_id = '/' + os.path.join(*save_id)

            # create a directory to save the ITRs in
            if not os.path.exists(save_id):
                os.makedirs(save_id)

            save_id = os.path.join(save_id, file_id)

            if verbose:
                print("n: {0}, filename: {1}, saved_id: {2}".format(n, file, save_id))

            # save ITR to file with given name
            print(save_id)
            np.savez(save_id, data=iads[n])

def generate_itr_files(lfd_params, model, dataset_mode, verbose=False, backbone="tsm"):

    # Create DataLoaders
    assert lfd_params.args.input_dtype in ["video", "itr"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if lfd_params.args.input_dtype == "video":
        from datasets.dataset_video import DatasetVideo as CustomDataset

    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, dataset_mode, verbose=True,
                            num_segments=lfd_params.input_frames)
    data_loader = create_dataloader(dataset, lfd_params, dataset_mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet

        # compute output
        itrs = net(obs)
        itrs = itrs.detach().cpu().numpy()

        for n, file in enumerate(filename):

            # format new save name
            save_id = file.split('/')
            file_id = save_id[-1] + ".npz"
            save_id = save_id[:save_id.index("frames")] + ["itrs_"+backbone] + save_id[save_id.index("frames") + 1:-1]
            save_id = '/' + os.path.join(*save_id)

            # create a directory to save the ITRs in
            if not os.path.exists(save_id):
                os.makedirs(save_id)

            save_id = os.path.join(save_id, file_id)

            if verbose:
                print("n: {0}, filename: {1}, saved_id: {2}".format(n, file, save_id))

            # save ITR to file with given name
            print(save_id)

            np.savez(save_id, data=itrs[n])



def generate_itr_files_gcn(lfd_params, model, dataset_mode, verbose=False, backbone="tsm"):
    # Create DataLoaders

    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, dataset_mode, verbose=True,
                            num_segments=lfd_params.input_frames)
    data_loader = create_dataloader(dataset, lfd_params, dataset_mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet

        # compute output
        print("filename:", filename)
        print("obs:", obs.shape)
        x = net(obs)
        node_x, edge_idx, edge_attr = x

        print("node_x:", node_x.shape)
        print("edge_idx:", edge_idx.shape)
        print("edge_attr:", edge_attr.shape)
        print("")

        for n, file in enumerate(filename):

            # format new save name
            save_id = file.split('/')
            file_id = save_id[-1]
            save_id = save_id[:save_id.index("iad_"+lfd_params.model.model_id)] + ["gcn_"+backbone] + save_id[save_id.index("iad_"+lfd_params.model.model_id) + 1:-1]
            save_id = '/' + os.path.join(*save_id)

            # create a directory to save the ITRs in
            if not os.path.exists(save_id):
                os.makedirs(save_id)

            save_id = os.path.join(save_id, file_id)

            if verbose:
                print("n: {0}, filename: {1}, saved_id: {2}".format(n, file, save_id))

            # save ITR to file with given name
            np.savez(save_id, x=node_x[0], edge_idx=edge_idx[0], edge_attr=edge_attr[0])
