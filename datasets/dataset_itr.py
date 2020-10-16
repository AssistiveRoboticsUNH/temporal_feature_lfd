import os
import numpy as np
from torch.utils.data import Dataset

from .utils import get_observation_list


class DatasetITR(Dataset):
    def __init__(self, root_path, mode, verbose=False):
        assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"
        self.mode = mode
        self.verbose = verbose

        assert os.path.exists(root_path), "ERROR: dataset_itr.py: Cannot locate path - " + root_path
        self.obs_dict = get_observation_list(root_path)

        # make data easily accessible
        self.data = []
        for k in self.obs_dict:
            self.data.extend(self.obs_dict[k])

    def parse_obs(self, filename):
        return np.load(filename)["data"]

    def __getitem__(self, index):
        filename = self.data[index]
        obs = self.parse_obs(filename)
        if self.verbose:
            return obs, filename
        return obs

    def __len__(self):
        return len(self.data)
