import torch
import os
import numpy as np

from datasets.utils import create_dataloader
from model.temporal.ditrl import DITRL_MaskFinder


from datasets.dataset_video import DatasetVideo as CustomDataset


def train_pipeline(lfd_params, model):

    # Create DataLoaders
    dataset = CustomDataset(lfd_params.file_directory, "train", num_segments=lfd_params.args.num_segments)
    data_loader = create_dataloader(dataset, lfd_params, "train", shuffle=False)

    # put model on GPU
    model.use_pipeline = False
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    # record mask and threshold values
    mask_and_threshold = DITRL_MaskFinder()

    for i, data_packet in enumerate(data_loader):
        obs, label = data_packet

        # compute output
        print("obs.shape:", obs.shape)
        activation_map = net(obs)

        print("activation_map.shape:", activation_map.shape)
        #activation_map = activation_map.view((-1, lfd_params.args.num_segments) + activation_map.size()[1:])
        activation_map = activation_map.detach().cpu().numpy()

        for iad in activation_map:
            mask_and_threshold.add_data(iad)

    mask, threshold = mask_and_threshold.gen_mask_and_threshold()
    model.use_pipeline = True
    model.pipeline.pipeline.preprocessing = False
    model.pipeline.pipeline.mask_idx = mask
    model.pipeline.pipeline.threshold_values = threshold

    print("threshold:", threshold)
    print("mask:", mask)
    #assert False

    for i, data_packet in enumerate(data_loader):
        obs, label = data_packet

        # compute output
        _ = net(obs)

    model.pipeline.fit_pipeline()

    model.pipeline.pipeline.preprocessing = True
    model.pipeline.pipeline.is_training = False

    return model


def generate_itr_files(lfd_params, model, dataset_mode, verbose=False, backbone="tsm"):

    # Create DataLoaders
    assert lfd_params.args.input_dtype in ["video", "itr"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if lfd_params.args.input_dtype == "video":
        from datasets.dataset_video import DatasetVideo as CustomDataset

    dataset = CustomDataset(lfd_params.file_directory, dataset_mode, verbose=True,
                            num_segments=lfd_params.args.num_segments)
    data_loader = create_dataloader(dataset, lfd_params, dataset_mode, shuffle=False)

    # put model on GPU
    #print("model.pipeline.is_training 0:", model.pipeline.is_training)
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    #print("model.pipeline.is_training 1:", model.pipeline.is_training)

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet

        # compute output
        #print("model.pipeline.is_training 2:", model.pipeline.is_training)
        itrs = net(obs)
        #print("model.pipeline.is_training 3:", model.pipeline.is_training)
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
    from torch_geometric.data import Data
    # Create DataLoaders
    assert lfd_params.args.input_dtype in ["video", "itr"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if lfd_params.args.input_dtype == "video":
        from datasets.dataset_video import DatasetVideo as CustomDataset

    dataset = CustomDataset(lfd_params.file_directory, dataset_mode, verbose=True,
                            num_segments=lfd_params.args.num_segments)
    data_loader = create_dataloader(dataset, lfd_params, dataset_mode, shuffle=False)

    # put model on GPU
    #print("model.pipeline.is_training 0:", model.pipeline.is_training)
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    #print("model.pipeline.is_training 1:", model.pipeline.is_training)

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet

        # compute output
        #print("model.pipeline.is_training 2:", model.pipeline.is_training)
        node_x, edge_idx, edge_attr = net(obs)
        #print("model.pipeline.is_training 3:", model.pipeline.is_training)
        #node_x, edge_idx, edge_atrr = data#.detach().cpu()
        #print("node_x:", node_x.shape)
        #print()
        #print()
        data = Data()

        #=edge_idx[0], edge_attr=edge_attr[0])

        for n, file in enumerate(filename):

            # format new save name
            save_id = file.split('/')
            file_id = save_id[-1] + ".gcn_data"
            save_id = save_id[:save_id.index("frames")] + ["gcn_"+backbone] + save_id[save_id.index("frames") + 1:-1]
            save_id = '/' + os.path.join(*save_id)

            # create a directory to save the ITRs in
            if not os.path.exists(save_id):
                os.makedirs(save_id)

            save_id = os.path.join(save_id, file_id)

            if verbose:
                print("n: {0}, filename: {1}, saved_id: {2}".format(n, file, save_id))

            print("data:", type(data))

            # save ITR to file with given name
            print(save_id)
            #np.save(data, save_id)
            np.savez(save_id, x=node_x[0], edge_idx=edge_idx[0], edge_attr=edge_attr[0])
