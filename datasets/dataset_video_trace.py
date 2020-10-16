import numpy as np
import os
import random

from .dataset_video import DatasetVideo, IMAGE_TMPL_DEF
NUM_TOTAL_ACTIONS = 4


class DatasetVideoTrace(DatasetVideo):
    def __init__(self, root_path, mode, trace_path, verbose=False, image_tmpl=IMAGE_TMPL_DEF, num_segments=3):
        super().__init__(root_path, mode, verbose=verbose, image_tmpl=image_tmpl, num_segments=num_segments)

        # open the file containing traces
        # ---
        assert os.path.exists(trace_path), "ERROR: dataset_video_trace.py: Cannot locate trace file at - " + trace_path

        # make the first 9/10 for training and save teh last 1/10 for evaluation
        self.traces = np.load(trace_path)
        chunk = -int(len(self.traces) / 10)
        if mode == "train":
            self.traces = self.traces[:chunk]
        else:
            self.traces = self.traces[chunk:]

        print("num_traces:", len(self.traces))

        # replace the trace values with filenames
        # ---
        self.obs_dict['n'] = ["None"]
        obs_labels = ['n', 'r', 'rr', 'rrr', 'g', 'gb', 'bg', 'b']
        self.data_shape = super().parse_obs(self.obs_dict['r'][0]).shape

        self.full_traces = []
        for obs, act in self.traces:
            obs_filename = [random.sample(self.obs_dict[obs_labels[o]], k=1)[0] for o in obs]
            self.full_traces.append((obs_filename, act))
        print("self.labelled_traces:", len(self.full_traces))

        # Create a corpus of shortened traces from the original length traces.
        # These are used to train the policy model
        # ---
        self.shrt_traces = []
        for obs, act in self.full_traces:
            for i in range(1, len(act)):
                self.shrt_traces.append((obs[:i], act[:i]))
        print("self.data:", len(self.shrt_traces))

    def parse_obs(self, filename_list):
        file_data = []
        for filename in filename_list:
            obs_data = np.zeros(self.data_shape)
            if filename != "None":
                obs_data = super().parse_obs(filename)
            file_data.append(obs_data)
        return np.stack(file_data).squeeze(axis=1)

    def parse_act(self, action_list):
        actions_out = np.zeros((len(action_list), NUM_TOTAL_ACTIONS), dtype=np.float32)
        for a in range(len(action_list)):
            actions_out[a, action_list[a]] = 1
        return actions_out

    def __getitem__(self, index):
        if self.mode == "train":
            obs_src, act_src = self.shrt_traces[index]
        else:
            obs_src, act_src = self.full_traces[index]

        obs = self.parse_obs(obs_src)
        act = self.parse_act(act_src)

        if self.verbose:
            return obs, act, obs_src, act_src
        return obs, act

    def __len__(self):
        if self.mode == "train":
            return len(self.shrt_traces)
        return len(self.full_traces)
