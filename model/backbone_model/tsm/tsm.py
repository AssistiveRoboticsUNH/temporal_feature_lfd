import sys, os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from .ops.models import TSN

class TSMWrapper(TSN):
    def __init__(self, 
            checkpoint_file,

            num_classes, 
            num_segments=3,
            training=False, 
            bottleneck_size=128,
            pretrained_checkpoint=False):

        super(TSMWrapper, self).__init__(num_classes, num_segments, 'RGB',
            base_model='resnet101', 
            consensus_type='avg',
            dropout=0.01,#1.0,#0.8,
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
        
        #print("self.base_model:", self.base_model)
        # apply Bottleneck and replace AvgPool with MaxPool

        self.bottleneck_size = bottleneck_size
        
        self.base_model.avgpool = nn.Sequential(
            nn.Conv2d(2048, self.bottleneck_size, (1,1)),
            #nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=1),
        )
        
        self.new_fc = nn.Identity() # this is not necessary but is helpful for narrowing down issues from a debugging perspective
        
        
        #load model
        print("Loading Backbone Model from: "+checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        if (pretrained_checkpoint):

            try:
                from collections import OrderedDict
                new_state_dict = OrderedDict()

                for k, v in checkpoint['state_dict'].items():
                    new_k = '.'.join(k.split('.')[2:])
                    if (".net" in new_k):
                        new_k = '.'.join(new_k.split('.')[:-2]+new_k.split('.')[-1:])
                    new_state_dict[new_k] = v
            except:
                print("ERROR: tsm.py: provided pretrain-checkpoint file "+checkpoint_file+" not formatted to work with model")
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in checkpoint.items():
                new_k = '.'.join(k.split('.')[1:])
                new_state_dict[new_k] = v
            
        '''
        print("var:")
        for k, v in new_state_dict.items():
            print(k)
        print("var done")
        '''
        self.base_model.load_state_dict(new_state_dict, strict=not training)

    def forward(self, inp):
        sample_len = 3 * self.new_length

        inp = inp.view((-1, sample_len) + inp.size()[-2:])
        
        base_out = self.base_model(inp)

        return base_out
       

    def save_model(self, filename, debug=False):
        if (debug):
            print("self.state_dict():")
            for k in self.state_dict().keys():
                print("\t"+k, self.state_dict()[k].shape )

        torch.save(self.state_dict(), filename )
        

