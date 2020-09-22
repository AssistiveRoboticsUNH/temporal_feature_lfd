import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self,
            checkpoint_file,
            num_classes,
            num_segments=3,
            training=False,
            bottleneck_size=128,
            pretrained_checkpoint=False):

        super().__init__()

        self.checkpoint_file = checkpoint_file
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.training = training
        self.bottleneck_size = bottleneck_size
        self.pretrained_checkpoint = pretrained_checkpoint

        hidden_layer = 4

        self.model = nn.Sequential(
            nn.Conv2d(3, hidden_layer, 1),
            nn.ReLU(),
            # nn.Conv2d(hidden_layer, self.bottleneck_size, (1, 1)),
            nn.AdaptiveMaxPool2d(output_size=1),
            )

        # load model
        print("Loading Backbone Model from: " + checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in checkpoint.items():
            new_k = '.'.join(k.split('.')[1:])
            new_state_dict[new_k] = v

        self.model.load_state_dict(new_state_dict, strict=not training)

    def forward(self, inp):
        sample_len = 3
        inp = inp.view((-1, sample_len) + inp.size()[-2:])
        print("inp.shape:", inp.shape)

        out = self.model(inp)
        print("out.shape:", out.shape)
        out = out.view([-1, 4])
        print("out.2shape:", out.shape)
        return out

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)