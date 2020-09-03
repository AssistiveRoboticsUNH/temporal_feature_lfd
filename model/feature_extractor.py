import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):   
	def __init__(self, lfd_params, is_training ):

		super().__init__()

		self.lfd_params = lfd_params 

		self.num_classes = lfd_params.num_actions
		self.use_aud =  lfd_params.use_aud
		self.is_training = is_training
		self.num_segments =  lfd_params.args.num_segments

		self.bottleneck_size = lfd_params.args.bottleneck_size

		# RGB MODEL 
		# ---

		# get the files to use with this model
		self.checkpoint_file = lfd_params.args.pretrain_modelname
		if (lfd_params.args.backbone_modelname):
			self.checkpoint_file = lfd_params.args.backbone_modelname

		# rgb net
		from .backbone_model.tsm.tsm import TSMWrapper as VisualFeatureExtractor
		self.rgb_net = VisualFeatureExtractor(
			self.checkpoint_file,
			self.num_classes, 
			training=self.is_training,
			num_segments=1#self.num_segments
			)


		"""
		#load model
        print("Loading Backbone Model from: "+checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        print("training:", training)
        if (training):
        #checkpoint = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in checkpoint['state_dict'].items():
                new_k = '.'.join(k.split('.')[2:])
                if (".net" in new_k):
                    new_k = '.'.join(new_k.split('.')[:-2]+new_k.split('.')[-1:])
                new_state_dict[new_k] = v
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in checkpoint.items():
                new_k = '.'.join(k.split('.')[1:])
                new_state_dict[new_k] = v

        #checkpoint['state_dict'] = mod_checkpoint

        '''
        print("vars:")
        for k in new_state_dict.keys():
            print(k)
        print("vars done")
        '''
        #print(new_state_dict['layer4.2.conv1.net.weight'])
        #print("self.base_model.state_dict():")
        #for k, v in self.base_model.state_dict().items():
        #    print(k)

        self.base_model.load_state_dict(new_state_dict, strict=not training)


		"""








		# parameter indicates that the backbone's features should be fixed
		# the following code prevents the modification of these layers by removing their gradient information
		if (lfd_params.args.backbone_modelname):
			for param in self.rgb_net.parameters():
				param.requires_grad = False
	

	# Defining the forward pass    
	def forward(self, rgb_x):
		return self.rgb_net(rgb_x)

	def save_model(self, debug=False):
		filename = self.lfd_params.generate_backbone_modelname()
		self.rgb_net.save_model(filename)
		
		print("Backbone model saved to: ", filename)
