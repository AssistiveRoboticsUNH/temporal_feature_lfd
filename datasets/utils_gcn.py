import os
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as DataLoaderGCN
import torch

def get_observation_list(root_path, mode):
    assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"

    root_path = os.path.join(root_path, mode)
    assert os.path.exists(root_path), "ERROR: Cannot locate path - " + root_path

    # get the ITR files
    obs_dict = {}
    for obs in os.listdir(root_path):
        #if obs in ['n', 'r', 'b', 'g']:
        all_obs_files = os.listdir(os.path.join(root_path, obs))
        #obs_dict[obs] = [os.path.join(*[root_path, obs, x]) for x in [all_obs_files[0]]]
        obs_dict[obs] = [os.path.join(*[root_path, obs, x]) for x in all_obs_files]
    return obs_dict




def create_dataloader(dataset, lfd_params, mode, shuffle=False):
    assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"

    return DataLoaderGCN(
        dataset,
        batch_size=1, #lfd_params.args.batch_size,
        shuffle=mode =="train" if shuffle is None else shuffle,
        num_workers=lfd_params.args.num_dl_workers,
        pin_memory=True)


def create_trace_dataloader(dataset, lfd_params, mode, shuffle=False):
    assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"

    return DataLoader(
        dataset,
        batch_size=1, #lfd_params.args.batch_size,
        shuffle=mode =="train" if shuffle is None else shuffle,
        num_workers=lfd_params.args.num_dl_workers,
        pin_memory=True)
'''
def create_trace_dataloader(dataset, lfd_params, mode, shuffle=False):
    assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"

    class CollaterTrace(object):
        def __init__(self, follow_batch):
            self.follow_batch = follow_batch

        def collate(self, batch):
            #elem = batch[0]
            return batch

        def __call__(self, batch):
            return self.collate(batch)

    class DataLoaderTrace(DataLoader):
        def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                     **kwargs):
            super().__init__(dataset, batch_size, shuffle,
                             collate_fn=CollaterTrace(follow_batch), **kwargs)

    return DataLoaderTrace(
        dataset,
        batch_size=1, #lfd_params.args.batch_size,
        shuffle=mode =="train" if shuffle is None else shuffle,
        num_workers=lfd_params.args.num_dl_workers,
        pin_memory=True)
'''
