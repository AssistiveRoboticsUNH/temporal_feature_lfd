import torch
import torch.nn as nn

from pytorch_i3d import InceptionI3d


class I3DWrapper(InceptionI3d):
    def __init__(self,
                 checkpoint_file,
                 num_classes,
                 num_segments=3,
                 training=False,
                 bottleneck_size=128,
                 pretrained_checkpoint=False):

        super(InceptionI3d, self).__init__(num_classes=num_classes, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.99)#dropout_keep_prob=0.5)

        # apply Bottleneck and replace AvgPool with MaxPool

        self.bottleneck_size = bottleneck_size

        self.base_model.avgpool = nn.Sequential(
            nn.Conv2d(2048, self.bottleneck_size, (1, 1)),
            nn.AdaptiveMaxPool2d(output_size=1),
        )

        # setting new_fc to the Identity is not necessary but helpful for clarity
        self.new_fc = nn.Identity()

        # load model
        print("Loading Backbone Model from: " + checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        if pretrained_checkpoint:

            try:
                from collections import OrderedDict
                new_state_dict = OrderedDict()

                for k, v in checkpoint['state_dict'].items():
                    new_k = '.'.join(k.split('.')[2:])
                    if ".net" in new_k:
                        new_k = '.'.join(new_k.split('.')[:-2] + new_k.split('.')[-1:])
                    new_state_dict[new_k] = v
            except:
                print("ERROR: tsm.py: provided pretrain-checkpoint file " + checkpoint_file +
                      " not formatted to work with model")
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in checkpoint.items():
                new_k = '.'.join(k.split('.')[1:])
                new_state_dict[new_k] = v

        self.base_model.load_state_dict(new_state_dict, strict=not training)

    def forward(self, inp):
        sample_len = 3 * self.new_length
        inp = inp.view((-1, sample_len) + inp.size()[-2:])

        base_out = self.base_model(inp)
        print("base_out:", base_out.shape)

        return base_out

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
