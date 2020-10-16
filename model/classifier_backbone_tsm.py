import torch.nn as nn

from .backbone_model.backbone_tsm import BackboneTSM
from .spatial.spatial_ext_linear import SpatialExtLinear


class ClassifierBackboneTSM(nn.Module):
    def __init__(self, lfd_params, filename,
                 spatial_train=False):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename

        self.backbone_filename = ".".join([self.filename, "backbone", "pt"])
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])

        # model sections
        self.backbone = BackboneTSM(lfd_params, is_training=spatial_train,
                                    filename=lfd_params.args.pretrain_modelname if spatial_train else self.backbone)
        self.spatial = SpatialExtLinear(lfd_params, is_training=spatial_train, filename=self.spatial_filename,
                                        input_size=2048)

        self.model = nn.Sequential(
            self.backbone,
            self.spatial,
        )

    # Defining the forward pass
    def forward(self, x):
        return self.model(x)

    def save_model(self):
        if self.spatial_train:
            self.backbone.save_model(self.backbone_filename)
            self.spatial.save_model(self.spatial_filename)
