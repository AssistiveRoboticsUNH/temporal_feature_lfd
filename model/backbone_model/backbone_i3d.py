import os
import torch
import torch.nn as nn

from .i3d.pytorch_i3d import InceptionI3d
from collections import OrderedDict
# is_training, filename = pretrain_filename
# not is_training, filename = backbone filename


class BackboneI3D(InceptionI3d):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 trim_model=False, output_size=400, num_segments=64, end_point=4):

        endpoints = ['Conv3d_1a_7x7', 'Conv3d_2c_3x3', 'Mixed_3b', 'Mixed_4f', 'Mixed_5c', 'Logits']
        endpoint = endpoints[end_point]

        super().__init__(num_classes=output_size, spatial_squeeze=True,
                 final_endpoint=endpoint, name='inception_i3d', in_channels=3, dropout_keep_prob=0.5)

        self.lfd_params = lfd_params
        self.filename = filename
        self.trim_model = trim_model
        self.num_segments = num_segments

        # load model parameters
        assert self.filename is not None, "ERROR: backbone_i3d.py: filename must be defined"
        self.load_model(self.filename, is_training)

    def forward(self, x):
        sample_len = 3
        x = x.view((-1, sample_len, self.num_segments) + x.size()[-2:])
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel
                if end_point == self._final_endpoint:
                    break

        if self.trim_model:
            return x

        x = torch.transpose(x, 2, 1)
        x = torch.max(x, dim=3)[0]
        x = torch.max(x, dim=3)[0]

        return x

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("BackboneI3D Linear model saved to: ", filename)

    def load_model(self, filename, is_training=True):
        assert os.path.exists(filename), "ERROR: backbone_i3d.py: Cannot locate saved model - " + filename

        # format the parameter list to match the variables. When using the pre-train dataset from TSM the variable
        # names need to be updated.
        checkpoint = torch.load(filename)
        new_state_dict = OrderedDict()

        for k, v in checkpoint.items():
            if "new_fc" not in k:
                new_k = '.'.join(k.split('.')[1:])
                new_state_dict[new_k] = v

        print("load:", type(checkpoint))
        self.load_state_dict(checkpoint, strict=not is_training)

        # do not allow the parameters to be changed when evaluating.
        if not is_training:
            for param in self.parameters():
                param.requires_grad = False
