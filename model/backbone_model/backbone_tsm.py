import os
import torch
import torch.nn as nn

from .tsm.ops.models import TSN
from collections import OrderedDict
# is_training, filename = pretrain_filename
# not is_training, filename = backbone filename


class BackboneTSM(TSN):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 trim_model=False, num_frames=3, output_size=2048):

        super().__init__(output_size, num_frames,  # num_classes, num_segments
                         'RGB',
                         base_model='resnet101',
                         consensus_type='avg',
                         dropout=0.5,
                         img_feature_dim=256,
                         partial_bn=True,
                         pretrain='imagenet',

                         is_shift=False,
                         shift_div=8,
                         shift_place='blockres',

                         new_length=None,
                         before_softmax=True,

                         fc_lr5=False,
                         temporal_pool=False,
                         non_local=False)

        self.lfd_params = lfd_params
        self.filename = filename

        # remove classification layers
        if trim_model:
            self.base_model.avgpool = nn.Identity()  # remove avgpool
            self.base_model.fc = nn.Identity()  # remove dropout
            self.new_fc = nn.Identity()  # setting new_fc to the Identity is not necessary but helpful for clarity

        # load model parameters
        assert self.filename is not None, "ERROR: backbone_tsm.py: filename must be defined"
        if is_training:
            self.load_model_pretrain(self.filename, self.fc)
        else:
            self.load_model(self.filename, self.fc)

    def forward(self, inp):
        sample_len = 3 * self.new_length
        inp = inp.view((-1, sample_len) + inp.size()[-2:])

        base_out = self.base_model(inp)

        return base_out

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("BackboneTSM Linear model saved to: ", filename)

    def load_model(self, filename, is_training=True):
        assert os.path.exists(filename), "ERROR: backbone_tsm.py: Cannot locate saved model - " + filename

        checkpoint = torch.load(filename)
        new_state_dict = OrderedDict()

        # format the parameter list to match the variables. When using the pre-train dataset from TSM the variable
        # names need to be updated.
        if is_training:
            try:
                for k, v in checkpoint['state_dict'].items():
                    new_k = '.'.join(k.split('.')[2:])
                    if ".net" in new_k:
                        new_k = '.'.join(new_k.split('.')[:-2]+new_k.split('.')[-1:])
                    new_state_dict[new_k] = v
            except():
                print("ERROR: backbone_tsm.py: provided pretrain-checkpoint file " + filename +
                      " not formatted to work with model")
        else:
            for k, v in checkpoint.items():
                new_k = '.'.join(k.split('.')[1:])
                new_state_dict[new_k] = v

        print("Loading BackboneTSM from: " + filename)
        self.base_model.load_state_dict(new_state_dict, strict=not is_training)

        # do not allow the parameters to be changed when evaluating.
        if not is_training:
            for param in self.base_model.parameters():
                param.requires_grad = False
