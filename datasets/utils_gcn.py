import os
from torch_geometric.data import DataLoader as DataLoaderGCN


def get_observation_list(lfd_params, root_path, mode):
    assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"

    root_path = os.path.join(root_path, mode)
    assert os.path.exists(root_path), "ERROR: Cannot locate path - " + root_path

    # get the ITR files
    obs_dict = {}
    legal_obs = lfd_params.application.obs_label_list.keys()
    for obs in os.listdir(root_path):
        if obs in legal_obs:
            all_obs_files = os.listdir(os.path.join(root_path, obs))
            obs_dict[obs] = [os.path.join(*[root_path, obs, x]) for x in all_obs_files]
    return obs_dict


def create_dataloader(dataset, lfd_params, mode, shuffle=False, batch_size=1):
    assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"

    return DataLoaderGCN(
        dataset,
        batch_size=batch_size, #lfd_params.args.batch_size,
        shuffle=mode =="train" if shuffle is None else shuffle,
        num_workers=lfd_params.dataloader_workers,
        pin_memory=True)

import random
def create_trace_dataloader(dataset, lfd_params, mode, shuffle=False):
    assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"

    class CustomDataLoader:
        def __init__(self, dataset, shuffle=True):
            self.dataset = dataset
            self.idx_list = list(range(len(self.dataset)))
            if shuffle:
                random.shuffle(self.idx_list)

        def __iter__(self):
            return (self.do_something(x) for x in self.idx_list)

        def do_something(self, x):
            return self.dataset[x]

        def __len__(self):
            return len(self.dataset)

    return CustomDataLoader(dataset, shuffle=mode =="train" if shuffle is None else shuffle)