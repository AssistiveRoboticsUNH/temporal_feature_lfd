import os
import torch.nn as nn

from .backbone_model.backbone_r21d import BackboneR21D
from .spatial.spatial_ext_linear import SpatialExtLinear


class ClassifierBackboneR21D(nn.Module):
    def __init__(self, lfd_params, filename,
                 spatial_train=False):
        super().__init__()
        self.lfd_params = lfd_params
        self.backbone_id = "r21d"

        # model filenames
        self.filename = filename

        # parts of model to train
        self.spatial_train = spatial_train

        self.backbone_filename = ".".join([self.filename, "backbone", "pt"])
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])

        # model sections
        pretrain_modelname = os.path.join(lfd_params.args.home_dir,
                                          "models/rgb_imagenet.pt")
        self.backbone = BackboneR21D(lfd_params, is_training=spatial_train,
                                    filename=pretrain_modelname if spatial_train else self.backbone_filename,
                                     trim_model=True)
        self.spatial = SpatialExtLinear(lfd_params, is_training=spatial_train, filename=self.spatial_filename,
                                        input_size=512, consensus="avg", dense_data=True)

    # Defining the forward pass
    def forward(self, x):
        print("classifier_backbone x.shape1:", x.shape)
        x = self.backbone(x)
        print("classifier_backbone x.shape2:", x.shape)
        x = self.spatial(x)
        print("classifier_backbone x.shape3:", x.shape)
        return x

    def save_model(self):
        if self.spatial_train:
            self.backbone.save_model(self.backbone_filename)

            self.spatial.save_model(self.spatial_filename)
