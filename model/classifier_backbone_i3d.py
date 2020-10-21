import os
import torch.nn as nn

from .backbone_model.backbone_i3d import BackboneI3D
from .spatial.spatial_ext_linear import SpatialExtLinear


class ClassifierBackboneI3D(nn.Module):
    def __init__(self, lfd_params, filename,
                 spatial_train=False):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename

        # parts of model to train
        self.spatial_train = spatial_train

        self.backbone_filename = ".".join([self.filename, "backbone", "pt"])
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])

        # model sections
        pretrain_modelname = os.path.join(lfd_params.args.home_dir,
                                          "models/rgb_imagenet.pt")
        self.backbone = BackboneI3D(lfd_params, is_training=spatial_train,
                                    filename=pretrain_modelname if spatial_train else self.backbone)
        self.spatial = SpatialExtLinear(lfd_params, is_training=spatial_train, filename=self.spatial_filename,
                                        input_size=2048, consensus="avg")

    # Defining the forward pass
    def forward(self, x):
        #print("classifier_backbone x.shape1:", x.shape)
        x = self.backbone(x)
        #print("classifier_backbone x.shape2:", x.shape)
        x = self.spatial(x)
        #print("classifier_backbone x.shape3:", x.shape)
        return x

    def save_model(self):
        if self.spatial_train:
            self.backbone.save_model(self.backbone_filename)
            self.spatial.save_model(self.spatial_filename)
