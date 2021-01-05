import numpy as np
import os
import random
import torch

from .dataset_gcn import DatasetGCN
from torch_geometric.data import Batch
NUM_TOTAL_ACTIONS = 4


class DatasetGCNTrace(DatasetGCN):
    def __init__(self, lfd_params,root_path, mode, trace_path, verbose=False, image_tmpl=None, num_segments=3, backbone="tsm", ablation=False):
        super().__init__(lfd_params,root_path, mode, backbone=backbone, verbose=verbose)

        # open the file containing traces
        # ---
        assert os.path.exists(trace_path), "ERROR: dataset_itr_trace.py: Cannot locate trace file at - " + trace_path

        # make the first 9/10 for training and save teh last 1/10 for evaluation
        self.traces = np.load(trace_path)
        chunk = -int(len(self.traces) / 10)
        if mode == "train":
            self.traces = self.traces[:chunk]
        else:
            self.traces = self.traces[chunk:]

        print("dataset_itr_trace.py: num_traces:", len(self.traces))

        # replace the trace values with filenames
        # ---
        # self.obs_dict['n'] = ["None"]
        obs_labels = ['n', 'r', 'rr', 'rrr', 'g', 'gb', 'bg', 'b']
        #self.data_shape = super().parse_obs(self.obs_dict['r'][0]).shape

        self.full_traces = []

        self.ablation = ablation
        if self.ablation:
            for o in self.obs_dict.keys():
                for video in self.obs_dict[o]:
                    obs_filename = [video, self.obs_dict['n'][0], self.obs_dict['n'][0]]
                    act = [0, 0, 0]
                    if o == 'r':
                        act[0] = 1
                    elif o == 'rr':
                        act[0] = 1
                        act[1] = 1
                    elif o == 'rrr':
                        act[0] = 1
                        act[1] = 1
                        act[2] = 1
                    elif o == 'g':
                        act[0] = 2
                    elif o == 'gb':
                        act[0] = 2
                        act[1] = 3
                    elif o == 'b':
                        act[0] = 3
                    elif o == 'bg':
                        act[0] = 3
                        act[1] = 2
                    self.full_traces.append((obs_filename, act))
        else:
            for obs, act in self.traces:
                obs_filename = [random.sample(self.obs_dict[obs_labels[o]], k=1)[0] for o in obs]
                self.full_traces.append((obs_filename, act))
            print("dataset_itr_trace.py: self.labelled_traces:", len(self.full_traces))

        # Create a corpus of shortened traces from the original length traces.
        # These are used to train the policy model
        # ---
        self.shrt_traces = []
        for obs, act in self.full_traces:
            for i in range(1, len(act)):
                self.shrt_traces.append((obs[:i], act[:i]))
        print("dataset_itr_trace.py: self.data:", len(self.shrt_traces))

    def parse_obs(self, filename_list):
        file_data = []
        for filename in filename_list:

            obs_data = super().parse_obs(filename)
            file_data.append(obs_data)

        #Batch.from_data_list(file_data)
        #for f in file_data:
        #    print(f)

        return file_data#Batch.from_data_list(file_data)#np.stack(file_data).squeeze(axis=1)

    def parse_act(self, action_list):
        actions_out = np.zeros((len(action_list), NUM_TOTAL_ACTIONS), dtype=np.float32)
        for a in range(len(action_list)):
            actions_out[a, action_list[a]] = 1
        return actions_out

    def __getitem__(self, index):
        if self.mode == "train" and not self.ablation:
            obs_src, act_src = self.shrt_traces[index]
        else:
            obs_src, act_src = self.full_traces[index]

        obs = self.parse_obs(obs_src)
        #print("dataset_gcn_trace.py obs:")
        #print(obs)
        #obs_x, obs_edge_index, obs_edge_attr = obs.x, obs.edge_index, obs.edge_attr
        act = self.parse_act(act_src)
        act = np.expand_dims(act, 0)
        act = torch.as_tensor(act)

        if self.verbose:
            print("obs_src:", obs_src)
            return obs, act, obs_src, act_src
        return obs, act

    def __len__(self):
        if self.mode == "train" and not self.ablation:
            return len(self.shrt_traces)
        return len(self.full_traces)
