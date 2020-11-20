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
    def forward(self, iad, sparse_iad_length=0):

        if self.use_gcn:
            activation_map = iad.detach().cpu().numpy()
            # activation_map = activation_map.detach().cpu().numpy()

            # pass data through D-ITR-L Pipeline
            node_x, edge_idx, edge_attr = [], [], []
            batch_num = activation_map.shape[0]
            for i in range(batch_num):
                node, edge_i, edge_a = self.pipeline.convert_activation_map_to_itr(activation_map[i])
                node_x.append(node)
                edge_idx.append(edge_i)
                edge_attr.append(edge_a)

                #print("itr:", itr)
                #itr_out.append(itr)
            node_x = np.array(node_x)
            edge_idx = np.array(edge_idx)
            edge_attr = np.array(edge_attr)

            # return ITRs
            return node_x, edge_idx, edge_attr  #torch.autograd.Variable(torch.from_numpy(itr_out).cuda())
        else:
            # detach activation map from pyTorch and convert to NumPy array
            activation_map = iad.detach().cpu().numpy()

            # pass data through D-ITR-L Pipeline
            out_list = []
            batch_num = activation_map.shape[0]
            for i in range(batch_num):
                iad = self.pipeline.convert_activation_map_to_iad(activation_map[i])
                if self.return_iad:
                    out_list.append(iad)
                else:
                    print("iad_length:", iad.shape)
                    iad_length = iad.shape[1]
                    sparse_map = self.pipeline.convert_iad_to_sparse_map(iad)
                    if self.return_vee:
                        vee = self.pipeline.sparse_map_to_iad(sparse_map, iad_length)
                        out_list.append(vee)
                    else:
                        itr = self.pipeline.convert_sparse_map_to_itr(sparse_map)
                        out_list.append(itr)

            out_list = np.array(out_list)

            # return ITRs
            return torch.autograd.Variable(torch.from_numpy(out_list).cuda())

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
