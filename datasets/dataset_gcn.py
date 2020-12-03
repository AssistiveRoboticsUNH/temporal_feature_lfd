import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


from .utils import get_observation_list


class DatasetGCN(Dataset):
    def __init__(self, lfd_params, root_path, mode, verbose=False, dataset_mode=None, image_tmpl=None, num_segments=3, backbone="tsm"):
        assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"
        self.mode = mode
        self.verbose = verbose

        if dataset_mode is None:
            dataset_mode = mode

        #root_path = os.path.join(root_path, "itrs")
        root_path = os.path.join(root_path, "gcn_"+backbone)
        assert os.path.exists(root_path), "ERROR: dataset_gcn.py: Cannot locate path - " + root_path
        self.obs_dict = get_observation_list(root_path, dataset_mode)
        self.obs_label_list = {"n": 0, "r": 1, "rr": 2, "rrr": 3, "g": 4, "gb": 5, "bg": 6, "b": 7}

        # make data easily accessible
        self.data = []
        for k in self.obs_dict:
            self.data.extend(self.obs_dict[k])

    def parse_obs(self, filename):
        data = np.load(filename)
        x, edge_idx, edge_attr = data['x'], data['edge_idx'], data['edge_attr']

        print("x:", x.shape)
        print("edge_idx:", edge_idx.shape)
        print("edge_attr:", edge_attr.shape)

        d = Data(x=x, edge_index=edge_idx)#, edge_attr=edge_attr)
        return d

    def get_label(self, filename):
        obs_name = filename.split('/')[-2]
        return self.obs_label_list[obs_name]

    def __getitem__(self, index):
        filename = self.data[index]
        obs = self.parse_obs(filename)
        print("filename:", filename)
        print("obs:", type(obs))

        label = self.get_label(filename)
        if self.verbose:
            return obs, label, filename
        return obs, label

    def __len__(self):
        return len(self.data)
