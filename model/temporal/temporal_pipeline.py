import os
import torch
import torch.nn as nn


import numpy as np
import pickle


class TemporalPipeline(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None, use_gcn=False):
        self.use_gcn = use_gcn
        if use_gcn:
            from .ditrl_gcn import DITRL_Pipeline
        else:
            from .ditrl import DITRL_Pipeline


        super().__init__()
        self.lfd_params = lfd_params

        # model filename
        self.filename = filename

        # constants params
        self.pipeline = DITRL_Pipeline(self.lfd_params.args.bottleneck_size, use_gcn=self.use_gcn)

        # define model vars
        if not is_training:
            assert self.filename is not None, \
                "ERROR: temporal_pipeline.py: filename must be defined when is_training is False"
            self.pipeline = self.load_model(self.filename)
            print("threshold:", self.pipeline.threshold_values)
            #assert False

        else:
            print("TemporalPipeline is training")

        self.pipeline.is_training = is_training

    # Defining the forward pass
    def forward(self, iad):

        if self.use_gcn:
            activation_map = iad.detach().cpu().numpy()
            # activation_map = activation_map.detach().cpu().numpy()

            # pass data through D-ITR-L Pipeline
            itr_out = []
            batch_num = activation_map.shape[0]
            for i in range(batch_num):
                itr = self.pipeline.convert_activation_map_to_itr(activation_map[i])
                print("itr:", itr)
                itr_out.append(itr)
            #itr_out = np.array(itr_out)

            # return ITRs
            return itr_out  #torch.autograd.Variable(torch.from_numpy(itr_out).cuda())
        else:
            # reshape iad to be [batch_size, num_frames, num_features]
            #activation_map = iad.view((-1, self.lfd_params.args.num_segments) + iad.size()[1:])

            # detach activation map from pyTorch and convert to NumPy array
            activation_map = iad.detach().cpu().numpy()
            #activation_map = activation_map.detach().cpu().numpy()

            # pass data through D-ITR-L Pipeline
            itr_out = []
            batch_num = activation_map.shape[0]
            for i in range(batch_num):
                itr = self.pipeline.convert_activation_map_to_itr(activation_map[i])
                itr_out.append(itr)
            itr_out = np.array(itr_out)

            # return ITRs
            return torch.autograd.Variable(torch.from_numpy(itr_out).cuda())

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.pipeline, f)
        print("TemporalPipeline saved to: ", filename)

    def load_model(self, filename):
        assert os.path.exists(filename), "ERROR: temporal_pipeline.py: Cannot locate saved model - "+filename

        print("Loading TemporalPipeline from: " + filename)
        with open(filename, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def fit_pipeline(self):
        if not self.use_gcn:
            self.pipeline.fit_tfidf()
