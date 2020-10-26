import os
import torch
import torch.nn as nn

from .i3d.pytorch_i3d import InceptionI3d
from collections import OrderedDict
# is_training, filename = pretrain_filename
# not is_training, filename = backbone filename


class BackboneI3D(InceptionI3d):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 trim_model=False, output_size=400, num_segments=64):

        endpoint = 'Mixed_5c'

        super().__init__(num_classes=output_size, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5)

        self.lfd_params = lfd_params
        self.filename = filename
        self.trim_model = trim_model
        self.num_segments = num_segments

        # load model parameters
        assert self.filename is not None, "ERROR: backbone_tsm.py: filename must be defined"
        self.load_model(self.filename, is_training)

    def forward(self, x):
        sample_len = 3 #* self.new_length
        #print("backbone x.shape1:", x.shape, sample_len)

        x = x.view((-1, sample_len, self.num_segments) + x.size()[-2:])
        #print("backbone x.shape2:", x.shape)

        #print("backbone x.shape3:", x.shape)

        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel

        #print("backbone x.shape4:", x.shape)
        if self.trim_model:
            return x

        x = self.avg_pool(x)
        #print("backbone x.shape5:", x.shape)
        x = torch.squeeze(x, 4)
        x = torch.squeeze(x, 3)
        x = torch.transpose(x, 2, 1)

        #print("backbone x.shape6:", x.shape)

        return x

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("BackboneTSM Linear model saved to: ", filename)

    def load_model(self, filename, is_training=True):
        assert os.path.exists(filename), "ERROR: backbone_tsm.py: Cannot locate saved model - " + filename

        checkpoint = torch.load(filename)
        new_state_dict = OrderedDict()

        # format the parameter list to match the variables. When using the pre-train dataset from TSM the variable
        # names need to be updated.
        print("filename:", filename)
        print("is_training:", is_training)

        self.load_state_dict(torch.load(self.filename))
        """
        if is_training:
            self.load_state_dict(torch.load(self.filename))
            #except():
            #    print("ERROR: backbone_tsm.py: provided pretrain-checkpoint file " + filename +
            #          " not formatted to work with model")
        else:
            self.load_state_dict(torch.load(self.filename))
        

        print("Loading BackboneTSM from: " + filename)
        self.base_model.load_state_dict(new_state_dict, strict=not is_training)
        """
        # do not allow the parameters to be changed when evaluating.
        if not is_training:
            for param in self.base_model.parameters():
                param.requires_grad = False
