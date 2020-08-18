import sys, os
import torch.nn as nn
import numpy as np
from PIL import Image

# include files from the backbone network source
#sys.path.append("/home/mbc2004/temporal-shift-module")
from .ops.models import TSN
from .ops import dataset_config
from .ops.transforms import *

class TSMWrapper(nn.Module):
    def __init__(self, 
            checkpoint_file, 
            num_classes, 
            max_length=8, 
            training=False, 
            bottleneck_size=128):
        super().__init__()

        self.bottleneck_size = bottleneck_size

        self.tsm = TSM( 
            checkpoint_file, 
            num_classes, 
            max_length, 
            training, 
            bottleneck_size)

    def forward(self, rgb_x):
        return self.tsm.net(rgb_x)


class TSM:
    def __init__(self, 
            checkpoint_file, 
            num_classes, 
            max_length=8, 
            training=False, 
            bottleneck_size=128,
            ):
        self.is_shift = None
        self.net = None
        self.arch = None
        self.num_classes = num_classes
        self.max_length = max_length

        # input variables
        this_test_segments = self.max_length

        #model variables
        self.is_shift, shift_div, shift_place = True, 8, 'blockres'
        self.arch = 'resnet101'
        modality = 'RGB'
        
        # dataset variables
        num_class, train_list, val_list, root_path, prefix = dataset_config.return_dataset('somethingv2', modality)
        print('=> shift: {}, shift_div: {}, shift_place: {}'.format(self.is_shift, shift_div, shift_place))

        # define model
        net = TSN(num_class, this_test_segments if self.is_shift else 1, modality,
                  base_model=self.arch,
                  consensus_type='avg',
                  img_feature_dim=256,
                  pretrain='imagenet',
                  is_shift=self.is_shift, shift_div=shift_div, shift_place=shift_place,
                  non_local='_nl' in checkpoint_file,
                  )

        # define bottleneck size
        self.bottleneck_size = bottleneck_size

        # Make any Necessary modifications to the model
        net.base_model.avgpool = nn.Sequential(
            nn.Conv2d(2048, self.bottleneck_size, (1,1)),
            nn.AdaptiveMaxPool2d(output_size=1),
        )
        
        # use trimmed net for generating IADs, use untrimed net when 
        # training the network

        if(training):
            net.new_fc = nn.Identity()
            #net.new_fc = nn.Linear(self.bottleneck_size, self.num_classes) # removed because I calulate this at higher level super calss
        else:
            net.consensus = nn.Identity()
            net.new_fc = nn.Identity()

        checkpoint = torch.load(checkpoint_file)
        if (not training):
            checkpoint = checkpoint.net.state_dict()
        else:
            checkpoint = checkpoint['state_dict']

        # Setup network to fine-tune the features that are already present
        # and to train those new layers I have defined
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                        'base_model.classifier.bias': 'new_fc.bias',
                        }
        for k, v in replace_dict.items():
            if v in base_dict:
                base_dict.pop(v)
            if k in base_dict:
                base_dict.pop(k)

        # load saved parameters into the file        
        net.load_state_dict(base_dict, strict=False)

        # define image modifications
        self.transform = torchvision.transforms.Compose([
                           torchvision.transforms.Compose([
                                GroupScale(net.scale_size),
                                GroupCenterCrop(net.scale_size),
                            ]),
                           #torchvision.transforms.Compose([ GroupFullResSample(net.scale_size, net.scale_size, flip=False) ]),
                           Stack(roll=(self.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(self.arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                           ])

        # place net onto GPU and finalize network
        self.model = net
        net = torch.nn.DataParallel(net.cuda())
        net.eval()

        # network variable
        self.net = net


    '''
    The following functions are used to generate IADs when given a single image at a time.
    They should be cleaned up and annotated after I have settled on a method for accomplishing this.
    This approach is currently required as the DataLoader approach does not generate file names. It
    is unclear if I will still need to hard code files in this manner.
    '''
    def open_file(self, csv_input, start_idx=0, batch_now=True):
        # replace this
        
        folder_name = csv_input['raw_path']
        assert os.path.exists(folder_name), "cannot find frames folder: "+folder_name
        files = os.listdir(folder_name)

        data = []
    
        for i in range(self.max_length):
            frame = start_idx+i
            if(frame < len(files)): 
                data.append( Image.open(os.path.join(folder_name, files[frame])).convert('RGB') ) 
            else:
                # fill out rest of video with blank data
                data.append( Image.new('RGB', (data[0].width, data[0].height)) )

        # process the frames
        data = self.transform(data)
        if (batch_now):
            return data.view(-1, self.max_length, 3, 256,256)
        return data.view(self.max_length, 3, 256,256)
        

    def process(self, csv_input):
        # replace this

        data_in = self.open_file(csv_input)
        length_ratio = csv_input['length']/float(self.max_length)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        # pass data through network to obtain activation maps
        # rst is not used and not need to store grads
        with torch.no_grad():

            rst = self.net(data_in)
            print("rst1:", rst.shape)
           
            # convert actvitaion from PyTorch to Numpy
            rst = rst.cpu().numpy()
            #rst = rst.reshape((-1, 128, 8,8))
            rst = rst.reshape((-1, self.bottleneck_size))

            print("rst2:", rst.shape)
            assert False

            # compress spatial dimensions
            rst = np.max(rst, axis=(2,3))
            rst = rst.T

        return rst, length_ratio

    
