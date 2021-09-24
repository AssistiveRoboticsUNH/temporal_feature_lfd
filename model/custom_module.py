import os
import torch
import torch.nn as nn


class CustomModule(nn.Module):
    def __init__(self, lfd_params, model_type, filename, input_size, output_size, is_training):

        super().__init__()
        self.lfd_params = lfd_params

        self.class_name = type(self).__name__
        self.filename = os.path.join(filename, ".".join(["model", model_type, "pt"]))

        # constants params
        self.input_size = input_size
        self.output_size = output_size

        # load model parameters
        if is_training:
            print(self.class_name + " is training")
        else:
            self.load_model(self.filename)

    # Defining the forward pass
    # expects [batch_size, frames, features]
    def forward(self, x):
        return x

    def save_model(self):
        torch.save(self.state_dict(), self.filename)
        print(self.class_name+" model saved to: "+self.filename)

    def load_model(self, filename):
        assert os.path.exists(filename), "ERROR: custom_module.py: Cannot locate saved model - " + filename

        print("Loading "+self.class_name+" from: " + filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint, strict=True)
        for param in self.parameters():
            param.requires_grad = False
