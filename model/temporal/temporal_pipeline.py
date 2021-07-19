import os
import torch
import torch.nn as nn


import numpy as np
import pickle


class TemporalPipeline(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 return_iad=False, return_vee=False, return_itr=True, use_gcn=False):

        self.return_iad = return_iad
        self.return_vee = return_vee
        self.return_itr = return_itr
        self.use_gcn = use_gcn

        if use_gcn:
            from .ditrl_gcn import DITRL_Pipeline
        else:
            from .ditrl import DITRL_Pipeline

        super().__init__()
        self.lfd_params = lfd_params

        # model filename
        self.filename = os.path.join(filename, ".".join(["model", "temporal_pipeline", "pt"]))

        # constants params
        self.pipeline = DITRL_Pipeline(self.lfd_params.model.bottleneck_size, use_gcn=self.use_gcn)

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
    def forward(self, iad, sparse_iad_length=0):

        #if self.use_gcn:
        activation_map = iad.detach().cpu().numpy()
        #print("iad.shape:", activation_map.shape)

        # pass data through D-ITR-L Pipeline
        node_x, edge_idx, edge_attr = [], [], []
        batch_num = activation_map.shape[0]
        for i in range(batch_num):
            node, edge_i, edge_a = self.pipeline.convert_activation_map_to_itr(activation_map[i])
            node_x.append(node)
            edge_idx.append(edge_i)
            edge_attr.append(edge_a)

        node_x = np.array(node_x)
        edge_idx = np.array(edge_idx)
        edge_attr = np.array(edge_attr)

        # return ITRs
        return node_x, edge_idx, edge_attr


    def save_model(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self.pipeline, f)
        print("TemporalPipeline saved to: ", self.filename)

    def load_model(self, filename):
        assert os.path.exists(filename), "ERROR: temporal_pipeline.py: Cannot locate saved model - "+filename

        print("Loading TemporalPipeline from: " + filename)
        with open(filename, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def fit_pipeline(self):
        if not self.use_gcn:
            self.pipeline.fit_tfidf()
