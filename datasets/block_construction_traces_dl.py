from torch.utils.data import DataLoader
from datasets.video_dataset import VideoDataset, ITRDataset
from torch.utils.data import Dataset
import os
import numpy as np
import random

IMAGE_TMPL_DEF = 'image_{:05d}.jpg'



class BlockConstructionTraceDataset(Dataset):
    def __init__(self, root_path, mode, trace_file):
        assert mode in ["train", "evaluation"], "ERROR: Mode param must be 'train' or 'evaluation'"
        self.mode = mode

        assert os.path.exists(root_path), "ERROR: Cannot locate path - " + root_path
        self.root_path = root_path

        # get the ITR files
        self.obs_dict = {}

        for obs in os.listdir(root_path):
            all_obs_files = os.listdir(os.path.join(root_path, obs))
            self.obs_dict[obs] = all_obs_files

        # open traces file and compose the data pairs
        self.traces = np.load(trace_file)
        if mode == "train":
            self.traces = self.traces[:-int(len(mode)/10)]
        else:
            self.traces = self.traces[-int(len(mode)/10):]

        obs_labels = ['n', 'r', 'rr', 'rrr', 'g', 'gb', 'bg', 'b']

        self.associated_traces = []
        for trace in self.traces:
            obs = trace[0]
            act = trace[1]

            obs_filename = []
            for o in obs:
                filename = random.sample(self.obs_dict[obs_labels[o]])
                obs_filename.append(filename)

            self.associated_traces.append((obs_filename, act))

        self.spaced_dataset = []
        for obs, act in self.associated_traces:
            for i in range(1, len(trace)):
                self.spaced_dataset.append((obs[:i], act[:i]))

    def parse_obs(self, filename_list):
        file_data = []
        for filename in filename_list:
            file_data = np.load(filename)["data"]
        return file_data

    def __getitem__(self, index):
        obs, act = self.spaced_dataset[index]
        return self.parse_obs(obs), act

    def __len__(self):
        return len(self.spaced_dataset)


def create_dataloader_itr(lfd_params, mode, shuffle=None, verbose=None):
    # setup path parameters
    assert mode in ["train", "validate", "evaluation"], \
        "ERROR: block_construction_dl.py: mode must be either 'train', 'validate', or 'evaluation'"
    is_training = (mode == "train")
    root_path = os.path.join(lfd_params.file_directory, mode)

    # create dataset
    dataset = BlockConstructionTraceDataset(root_path,
                                          mode,
                                          trace_file="/home/mbc2004/BlockConstruction/traces.npy")

    # create dataloader
    return DataLoader(
        dataset,
        batch_size=lfd_params.args.batch_size,
        shuffle=is_training if shuffle is None else shuffle,
        num_workers=lfd_params.args.num_dl_workers,
        pin_memory=True)
