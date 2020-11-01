import os
from torch.utils.data import DataLoader

def get_observation_list(root_path, mode):
    assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"

    root_path = os.path.join(root_path, mode)
    assert os.path.exists(root_path), "ERROR: Cannot locate path - " + root_path

    # get the ITR files
    obs_dict = {}
    for obs in os.listdir(root_path):
        if obs in ['n', 'r', 'b', 'g']:
            all_obs_files = os.listdir(os.path.join(root_path, obs))
            obs_dict[obs] = [os.path.join(*[root_path, obs, x]) for x in all_obs_files]
    return obs_dict


def create_dataloader(dataset, lfd_params, mode, shuffle=False):
    assert mode in ["train", "evaluation"], "ERROR: dataset_itr.py: Mode param must be 'train' or 'evaluation'"

    return DataLoader(
        dataset,
        batch_size=lfd_params.args.batch_size,
        shuffle=mode =="train" if shuffle is None else shuffle,
        num_workers=lfd_params.args.num_dl_workers,
        pin_memory=True)
