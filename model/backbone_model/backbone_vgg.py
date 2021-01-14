import os
import torch
import torch.nn as nn

from collections import OrderedDict
import torchvision

class BackboneVGG(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 trim_model=True, output_size=2048):
        super().__init__()

        self.base_model = torchvision.models.vgg16(pretrained=is_training)

        self.lfd_params = lfd_params
        self.filename = filename
        self.trim_model = trim_model

        print("trim_model:", trim_model)
        print("self.base_model.avgpool:", self.base_model.avgpool)

        # remove classification layers
        if self.trim_model:
            self.base_model.avgpool = nn.Identity()  # remove avgpool
        self.base_model.classifier = nn.Identity()  # remove dropout

        # load model parameters
        assert self.filename is not None, "ERROR: backbone_tsm.py: filename must be defined"
        if not is_training:
            self.load_model(self.filename, is_training)

    def forward(self, x):
        sample_len = 3
        print("backbone x.shape1:", x.shape, sample_len)

        x = x.view((-1, sample_len) + x.size()[-2:])
        print("backbone x.shape2:", x.shape)

        x = self.base_model.features(x)#self.base_model.forward(x)
        print("backbone x.shape3:", x.shape)
        x = torch.max(x, 2)[0]
        x = torch.max(x, 2)[0]
        print("backbone x.shape3.5:", x.shape)

        x = x.view((-1, self.lfd_params.args.num_segments) + x.size()[1:])
        print("backbone x.shape4:", x.shape)

        return x

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("BackboneVGG model saved to: ", filename)

    def load_model(self, filename, is_training=True):
        assert os.path.exists(filename), "ERROR: backbone_tsm.py: Cannot locate saved model - " + filename

        checkpoint = torch.load(filename)
        new_state_dict = OrderedDict()


        # format the parameter list to match the variables. When using the pre-train dataset from TSM the variable
        # names need to be updated.
        print("filename:", filename)
        print("is_training:", is_training)


        for k, v in checkpoint.items():
            if "new_fc" not in k:
                new_k = '.'.join(k.split('.')[1:])
                new_state_dict[new_k] = v


        print("Loading BackboneTSM from: " + filename)
        #self.base_model.load_state_dict(checkpoint, strict=not is_training)
        self.base_model.load_state_dict(new_state_dict, strict=not is_training)

        # do not allow the parameters to be changed when evaluating.
        if not is_training:
            for param in self.base_model.parameters():
                param.requires_grad = False


