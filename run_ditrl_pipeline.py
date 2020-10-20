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
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    # record mask and threshold values
    mask_and_threshold = DITRL_MaskFinder()

    for i, data_packet in enumerate(data_loader):
        obs, label = data_packet

        # compute output
        activation_map = net(obs, trim_after_spatial=True)
        activation_map = activation_map.view((-1, lfd_params.args.num_segments) + activation_map.size()[1:])
        activation_map = activation_map.detach().cpu().numpy()

        for iad in activation_map:
            mask_and_threshold.add_data(iad)

    mask, threshold = mask_and_threshold.gen_mask_and_threshold()
    model.pipeline.preprocessing = False
    model.pipeline.mask_idx = mask
    model.pipeline.threshold_values = threshold

    for i, data_packet in enumerate(data_loader):
        obs, label, filenames = data_packet

        # compute output
        _ = net(obs, trim_after_pipeline=True)

    model.pipeline.fit_pipeline()

    model.pipeline.preprocessing = True
    model.pipeline.is_training = False
    return model


def generate_itr_files(lfd_params, model, dataset_mode, verbose=False):

    # Create DataLoaders
    assert lfd_params.args.input_dtype in ["video", "itr"], "ERROR: run_videos.py: input_dtype must be 'video' or 'itr'"

    if lfd_params.args.input_dtype == "video":
        from datasets.dataset_video import DatasetVideo as CustomDataset
    else:
        from datasets.dataset_itr import DatasetITR as CustomDataset
    dataset = CustomDataset(lfd_params.file_directory, dataset_mode, verbose=True,
                            num_segments=lfd_params.args.num_segments)
    data_loader = create_dataloader(dataset, lfd_params, dataset_mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet

        # compute output
        itrs = net(obs, trim_after_pipeline=True)
        itrs = itrs.detach().cpu().numpy()

        for n, file in enumerate(filename):

            # format new save name
            save_id = file.split('/')
            file_id = save_id[-1] + ".npz"
            save_id = save_id[:save_id.index("frames")] + ["itrs"] + save_id[save_id.index("frames") + 1:-1]
            save_id = '/' + os.path.join(*save_id)

            # create a directory to save the ITRs in
            if not os.path.exists(save_id):
                os.makedirs(save_id)

            save_id = os.path.join(save_id, file_id)

            if verbose:
                print("n: {0}, filename: {1}, saved_id: {2}".format(n, file, save_id))

            # save ITR to file with given name
            print(save_id)
            #np.savez(save_id, data=itrs[n])

