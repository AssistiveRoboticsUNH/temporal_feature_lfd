import numpy as np
import os
import random

from .dataset_iad import DatasetIAD
NUM_TOTAL_ACTIONS = 4


class DatasetIADTrace(DatasetIAD):
    def __init__(self, lfd_params, root_path, mode, trace_path, verbose=False,
                 image_tmpl=None, num_segments=3, backbone="tsm", ablation=False, eval=True):
        super().__init__(lfd_params, root_path, mode, backbone=backbone, verbose=verbose)

        # open the file containing traces
        # ---
        assert os.path.exists(trace_path), "ERROR: dataset_iad_trace.py: Cannot locate trace file at - " + trace_path

        # make the first 9/10 for training and save teh last 1/10 for evaluation
        self.traces = np.load(trace_path)
        chunk = -int(len(self.traces) / 10)
        if mode == "train":
            self.traces = self.traces[:chunk]
        else:
            self.traces = self.traces[chunk:]

        self.eval = eval

        # replace the trace values with filenames
        # ---
        obs_labels = list(lfd_params.application.obs_label_list.keys())
        self.data_shape = super().parse_obs(self.obs_dict['r'][0]).shape

        self.full_traces = []

        self.ablation = ablation
        if self.ablation:
            for o in self.obs_dict.keys():
                for video in self.obs_dict[o]:
                    obs_filename = [video] + (random.sample(self.obs_dict['n'], 2))
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

        # Create a corpus of shortened traces from the original length traces.
        # These are used to train the policy model
        # ---
        self.shrt_traces = []
        for obs, act in self.full_traces:
            for i in range(1, len(act)+1):
                self.shrt_traces.append((obs[:i], act[:i]))

    def parse_obs(self, filename_list):
        file_data = []
        for filename in filename_list:
            obs_data = super().parse_obs(filename)
            file_data.append(obs_data)
        return np.stack(file_data)

    def parse_act(self, action_list):
        actions_out = np.zeros((len(action_list), NUM_TOTAL_ACTIONS), dtype=np.float32)
        for a in range(len(action_list)):
            actions_out[a, action_list[a]] = 1
        return actions_out

    def __getitem__(self, index):
        if not self.eval:
            obs_src, act_src = self.shrt_traces[index]
        else:
            obs_src, act_src = self.full_traces[index]

        obs = self.parse_obs(obs_src)
        act = self.parse_act(act_src)

        if self.verbose:
            return obs, act, obs_src, act_src
        return obs, act

    def __len__(self):
        if not self.eval:
            return len(self.shrt_traces)
        return len(self.full_traces)
