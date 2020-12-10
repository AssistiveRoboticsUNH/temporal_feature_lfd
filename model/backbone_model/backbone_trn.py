import os
import torch
import torch.nn as nn

from .trn.models import TSN
from collections import OrderedDict
# is_training, filename = pretrain_filename
# not is_training, filename = backbone filename


class BackboneTRN(TSN):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 trim_model=False, output_size=2048):

        super().__init__(output_size, lfd_params.args.num_segments,  # num_classes, num_segments
                         'RGB',
                         base_model='resnet101',
                         consensus_type='avg',
                         dropout=0.5,
                         img_feature_dim=256,
                         partial_bn=True,

                         new_length=None,
                         before_softmax=True)

        self.lfd_params = lfd_params
        self.filename = filename
        self.trim_model = trim_model

        # remove classification layers
        if self.trim_model:
            self.base_model.avgpool = nn.Identity()  # remove avgpool
        self.base_model.fc = nn.Identity()  # remove dropout
        self.new_fc = nn.Identity()  # setting new_fc to the Identity is not necessary but helpful for clarity

        # load model parameters
        assert self.filename is not None, "ERROR: backbone_trn.py: filename must be defined"
        self.load_model(self.filename, is_training)

    def forward(self, x):
        sample_len = 3 * self.new_length
        #print("backbone x.shape1:", x.shape, sample_len)

        x = x.view((-1, sample_len) + x.size()[-2:])
        #print("backbone x.shape2:", x.shape)

        #print("x in:")
        #print(x)

        x = self.base_model(x)
        #print("backbone x.shape3:", x.shape)


        print("self.base_model")
        print(self.base_model)

        x = x.view((-1, self.lfd_params.args.num_segments) + x.size()[1:])

        return x

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("BackboneTRN Linear model saved to: ", filename)

    def load_model(self, filename, is_training=True):
        assert os.path.exists(filename), "ERROR: backbone_trn.py: Cannot locate saved model - " + filename

        checkpoint = torch.load(filename)
        new_state_dict = OrderedDict()

        # format the parameter list to match the variables. When using the pre-train dataset from TSM the variable
        # names need to be updated.
        print("filename:", filename)
        print("is_training:", is_training)

        if is_training:
            try:
                for k, v in checkpoint['state_dict'].items():
                    new_k = '.'.join(k.split('.')[2:])
                    if ".net" in new_k:
                        new_k = '.'.join(new_k.split('.')[:-2]+new_k.split('.')[-1:])
                    new_state_dict[new_k] = v
            except():
                print("ERROR: backbone_trn.py: provided pretrain-checkpoint file " + filename +
                      " not formatted to work with model")
        else:
            for k, v in checkpoint.items():
                if "new_fc" not in k:
                    new_k = '.'.join(k.split('.')[1:])
                    new_state_dict[new_k] = v

        print("Loading BackboneTRN from: " + filename)
        self.base_model.load_state_dict(new_state_dict, strict=not is_training)

        # do not allow the parameters to be changed when evaluating.
        if not is_training:
            for param in self.base_model.parameters():
                param.requires_grad = False
