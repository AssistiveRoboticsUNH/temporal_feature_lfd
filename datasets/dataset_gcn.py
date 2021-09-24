import os
import numpy as np
import torch

from torch_geometric.data import Dataset
from torch_geometric.data import Data

from .utils_gcn import get_observation_list


class DatasetGCN(Dataset):
    def __init__(self, lfd_params, root_path, mode, verbose=False, dataset_mode=None,
                 image_tmpl=None, num_segments=3, backbone="tsm"):
        assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"
        self.mode = mode
        self.verbose = verbose

        if dataset_mode is None:
            dataset_mode = mode

        root_path = os.path.join(root_path, "gcn_"+backbone)
        assert os.path.exists(root_path), "ERROR: dataset_gcn.py: Cannot locate path - " + root_path
        self.obs_dict = get_observation_list(lfd_params, root_path, dataset_mode)
        self.obs_label_list = lfd_params.application.obs_label_list

        # make data easily accessible
        self.data = []
        for k in self.obs_dict:
            self.data.extend(self.obs_dict[k])

    def parse_obs(self, filename):
        data = np.load(filename)

        x, edge_idx, edge_attr = data['x'], data['edge_idx'], data['edge_attr']

        if x.shape[0] == 0:
            x = np.zeros((1, x.shape[1]))
            edge_idx = np.array([[0, 0]]).T
            edge_attr = np.array([0])

        x = torch.as_tensor(x)
        edge_idx = torch.as_tensor(edge_idx).long()
        edge_attr = torch.as_tensor(edge_attr).long()

        d = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)
        return d

    def get_label(self, filename):
        obs_name = filename.split('/')[-2]
        return self.obs_label_list[obs_name]

    def __getitem__(self, index):
        filename = self.data[index]
        label = self.get_label(filename)

        obs = self.parse_obs(filename)

        if self.verbose:
            return obs, label, filename
        return obs, label

    def __len__(self):
        return len(self.data)
