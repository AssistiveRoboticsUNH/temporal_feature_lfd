import os
import torch
import torch.nn as nn

from .iad_2_itr_layers import IAD2MaskedIAD, MaskedIAD2ITR


class TemporalPipeline(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None):

        super().__init__()
        self.lfd_params = lfd_params

        # model filename
        self.filename = os.path.join(filename, ".".join(["model", "temporal_pipeline_custom", "pt"]))

        # constants params
        self.iad2mask = IAD2MaskedIAD()
        self.mask2itr = MaskedIAD2ITR()

        # define model vars
        if not is_training:
            assert self.filename is not None, \
                "ERROR: temporal_pipeline_custom.py: filename must be defined when is_training is False"
            self.load_model(self.filename)
        else:
            print("TemporalPipeline is training")

    # Defining the forward pass
    def forward(self, iad):

        mask = self.iad2mask(iad)
        itr = self.mask2itr(mask)

        # return ITRs
        return itr

    def save_model(self):
        torch.save(self.state_dict(), self.filename)
        print("TemporalExtPipeline model saved to: ", self.filename)

    def load_model(self, filename):
        assert os.path.exists(filename), "ERROR: temporal_ext_linear.py: Cannot locate saved model - "+filename

        print("Loading TemporalExtLinear from: " + filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint, strict=True)
        for param in self.parameters():
            param.requires_grad = False
