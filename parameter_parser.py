import argparse
import os
import datetime

ROOT_DIR = "/home/mbc2004/"
SEGMENTS = 16
EPOCHS = 50
ALPHA = 0.0001 # 0.0001
BATCH = 1  # 3


class Parameters:
    def __init__(self, args):

        self.args = args

        self.create_dataloader = None
        self.file_directory = None
        self.num_actions = None
        self.num_hidden_state_params = None
        self.use_aud = None
        self.checkpoint_file = None
        self.trained_checkpoint_file = None

        self.debug = False

        if self.args.app == "bi":
            self.setup_social_greeting()
        elif self.args.app == "bs":
            self.setup_block_stacking()

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        #if self.args.save_id != "":
        #    self.locate_model_files_from_save_id()
        #else:
        #    self.generate_save_id()

        if self.debug:
            print(self.args)

    def use_itrs(self, value=True):

        if value:
            ''' 
            if self.args.app == "bi":
                from obsolete_files.datasets_old.social_greeting_dl import create_dataloader_itr
            elif self.args.app == "bs":
                from obsolete_files.datasets_old.block_construction_dl import create_dataloader_itr
            '''
            self.create_dataloader = create_dataloader_itr

            self.file_directory = self.file_directory.split('/')[:-2] + ["itrs"]
            self.file_directory = '/' + os.path.join(*self.file_directory)

            print("self.file_directory:", self.file_directory)
        else:
            if self.args.app == "bi":
                self.setup_social_greeting()
            elif self.args.app == "bs":
                self.setup_block_stacking()

    def clear_model_files_from_save_id(self):
        self.args.backbone_modelname = False
        self.args.ext_modelname = False
        self.args.policy_modelname = False

    def locate_model_files_from_save_id(self):
        filename = os.path.join(self.args.model_dir, "saved_model_"+self.args.save_id+"."+"backbone"+".pt")
        if os.path.exists(filename):
            print("file found: ", filename)
            self.args.backbone_modelname = filename
        elif not self.args.backbone_modelname:
            self.args.backbone_modelname = False

        filename = os.path.join(self.args.model_dir, "saved_model_" + self.args.save_id + "." + "ditrl" + ".pk")
        if os.path.exists(filename):
            print("file found: ", filename)
            self.args.ditrl_modelname = filename
        elif not self.args.ditrl_modelname:
            self.args.ditrl_modelname = False

        filename = os.path.join(self.args.model_dir, "saved_model_"+self.args.save_id+"."+"ext"+".pt")
        if os.path.exists(filename):
            print("file found: ", filename)
            self.args.ext_modelname = filename
        elif not self.args.ext_modelname:
            self.args.ext_modelname = False

        filename = os.path.join(self.args.model_dir, "saved_model_"+self.args.save_id+"."+"policy"+".pt")
        if os.path.exists(filename):
            print("file found: ", filename)
            self.args.policy_modelname = filename
        elif not self.args.policy_modelname:
            self.args.policy_modelname = False

    def setup_social_greeting(self):
        print("Loading social_greeting_dl")
        self.file_directory = os.path.join(ROOT_DIR, "datasets/SocialGreeting/frames/")
        self.num_actions = 3
        self.num_hidden_state_params = 1

        self.use_aud = False
        self.checkpoint_file = os.path.join(ROOT_DIR, "models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth")
        self.trained_checkpoint_file = os.path.join(ROOT_DIR, "models/social_greeting_tsm.pth")

        from obsolete_files.datasets_old.social_greeting_dl import create_dataloader
        self.create_dataloader = create_dataloader

    def setup_block_stacking(self):
        print("Loading block_construction_dl")
        self.file_directory = os.path.join(ROOT_DIR, "datasets/BlockConstruction/")
        self.num_actions = 7
        self.num_observations = 7
        self.num_hidden_state_params = 1

        self.use_aud = False
        self.checkpoint_file = os.path.join(ROOT_DIR, "models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth")
        self.trained_checkpoint_file = os.path.join(ROOT_DIR, "models/block_construction_tsm.pth")

        from obsolete_files.datasets_old.block_construction_dl import create_dataloader
        self.create_dataloader = create_dataloader


    def generate_save_id(self):
        if self.args.save_id == "":
            current_dt = datetime.datetime.now()
            use_ditrl = "ditrl_" if self.args.use_ditrl else ""
            use_trim = "trim_" if self.args.trim_model else ""
            self.args.save_id = self.args.app + "_" + use_ditrl+use_trim + current_dt.strftime("%Y-%m-%d_%H-%M-%S")

    def generate_modelname(self, section="null", suffix=".pt"):
        self.generate_save_id()
        return os.path.join(self.args.model_dir, "saved_model_"+self.args.save_id+"."+section+suffix)

    def generate_backbone_modelname(self):
        return self.generate_modelname(section="backbone")

    def generate_ext_modelname(self):
        return self.generate_modelname(section="ext")

    def generate_ditrl_modelname(self):
        return self.generate_modelname(section="ditrl", suffix=".pkl")

    def generate_ditrl_ext_modelname(self):
        return self.generate_modelname(section="ditrl")

    def generate_policy_modelname(self):
        return self.generate_modelname(section="policy")

    def print_params(self):
        print("""
            -------------
            File parameters
            -------------
            """)

        print("spatial/temporal: {:s}".format('TEMPORAL' if self.args.use_ditrl else 'SPATIAL'))
        print("model size: {:s}".format('FULL POLICY' if self.args.use_ditrl else 'JUST MODEL'))

        print("num segments: {:d}".format(self.args.num_segments))
        print("gaussian value: {:d}".format(self.args.gaussian_value))



# Constants

# system parameters
GPUS = [0]
DATALOADER_WORKERS = 8

# optimization parameters
BATCH_SIZE = 1
EPOCHS = 50
LR = 0.0001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

# image pre-processing parameters
GAUSSIAN_VALUE = 0

# directory locations
HOME_DIR = "/home/mbc2004"

BLOCK_STACKING_DIR = "/home/mbc2004/datasets/BlockConstruction"
BLOCK_STACKING_TRACE_FILE = "traces6.npy"

BASE_MODEL_DIR = "base_models"
MODEL_SAVE_DIR = "saved_models"

# input parameters
INPUT_FRAMES = 64



def default_model_params():
    class Params:
        def __init__(self,
                     gpus=GPUS,
                     dataloader_workers=DATALOADER_WORKERS,

                     batch_size=BATCH_SIZE,
                     epochs=EPOCHS,
                     lr=LR,
                     weight_decay=WEIGHT_DECAY,
                     momentum=MOMENTUM,

                     gaussian_value=GAUSSIAN_VALUE,

                     home_dir=HOME_DIR,
                     file_directory=BLOCK_STACKING_DIR,
                     trace_file=BLOCK_STACKING_TRACE_FILE,
                     model_save_dir=MODEL_SAVE_DIR,
                     base_model_dir=BASE_MODEL_DIR,

                     input_frames=INPUT_FRAMES
                     ):
            self.gpus = gpus
            self.dataloader_workers = dataloader_workers

            self.batch_size = batch_size
            self.epochs = epochs  # number of epochs to run experiments for
            self.lr = lr  #
            self.weight_decay = weight_decay # ?
            self.momentum = momentum  # ?

            self.gaussian_value = gaussian_value

            self.home_dir = home_dir

            self.file_directory = file_directory
            self.trace_file = os.path.join(self.file_directory, trace_file)

            self.base_model_dir = base_model_dir
            self.model_save_dir = model_save_dir

            self.input_frames = input_frames

            self.model = "unassigned"

        class ModelDef:
            def __init__(self, model_id, bottleneck_size, original_size, iad_frames, spatial_size,
                         backbone_class, pretrain_model_name=None, save_id=None, end_point=-1):
                self.end_point = end_point

                self.model_id = model_id
                self.bottleneck_size = bottleneck_size
                self.original_size = original_size[self.end_point]
                self.iad_frames = iad_frames[self.end_point]
                self.spatial_size = spatial_size

                self.backbone_class = backbone_class
                self.pretrain_model_name = pretrain_model_name
                self.save_id = save_id

        def set_model_params(self, model_id, end_point=-1):
            from enums import Backbone

            pretrain_model_name = None
            #save_id = "classifier_bottleneck_r21d0"

            if model_id == Backbone.TSM:
                from model.backbone_model.backbone_tsm import BackboneTSM as backbone_class
                pretrain_model_name = os.path.join(self.home_dir,
                    "models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth")
                #save_id = "classifier_bottleneck_tsm3"
                save_id = "c_backbone_tsm_0"
                self.model = self.ModelDef("tsm", 16, [2048], [64], 7, backbone_class,
                                           pretrain_model_name=pretrain_model_name,
                                           save_id=save_id)

            elif model_id == Backbone.TRN:
                from model.backbone_model.backbone_trn import BackboneTRN as backbone_class
                pretrain_model_name = os.path.join(self.home_dir,
                    "models/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar")
                self.model = self.ModelDef("trn", 16, [2048], [64], 7, backbone_class,
                                           pretrain_model_name=pretrain_model_name)

            elif model_id == Backbone.WRN:
                from model.backbone_model.backbone_wrn import BackboneWideResNet as backbone_class
                save_id = "classifier_bottleneck_wrn1"
                self.model = self.ModelDef("wrn", 16, [2048], [64], 7, backbone_class,
                                           save_id=save_id)

            elif model_id == Backbone.VGG:
                from model.backbone_model.backbone_vgg import BackboneVGG as backbone_class
                save_id = "classifier_bottleneck_vgg0"
                self.model = self.ModelDef("vgg", 32, [512], [64], 7, backbone_class,
                                           save_id=save_id)

            elif model_id == Backbone.I3D:
                original_size = [64, 192, 256, 832, 1024, 1024]
                iad_frames = [32, 32, 32, 16, 8, 8]

                from model.backbone_model.backbone_i3d import BackboneI3D as backbone_class
                pretrain_model_name = os.path.join(self.home_dir,
                    "models/rgb_imagenet.pt")
                save_id = "c_backbone_i3d_0"

                self.model = self.ModelDef("i3d", 16, original_size, iad_frames, 7, backbone_class,
                                           pretrain_model_name=pretrain_model_name,
                                           save_id=save_id,
                                           end_point=end_point)

    return Params()



def default_model_args(use_ditrl=False,
                       trim_model=False,
                       save_id="",
                       backbone_model="/home/mbc2004/models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth",
                       num_segments=SEGMENTS,
                       gaussian_value=0,
                       input_dtype="video",
                       epochs=EPOCHS,
                       log_dir="analysis/fig",
                       batch_size=1,
                       bottleneck_size=16,
                       dense_sample=False,
                       dense_rate=6):
    parser = argparse.ArgumentParser(description='Generate IADs from input files')

    # model command line args
    parser.set_defaults(app='bs')

    # whether the model should use D-ITR-L or not
    parser.set_defaults(use_ditrl=use_ditrl)
    parser.set_defaults(trim_model=trim_model)

    parser.set_defaults(bottleneck_size=bottleneck_size)

    # whether the model is being trained
    parser.set_defaults(model_dir="saved_models")
    parser.set_defaults(output_dir="csv_output")

    parser.set_defaults(save_id=save_id)
    parser.set_defaults(pretrain_modelname=backbone_model)
    parser.set_defaults(backbone_modelname=False)
    parser.set_defaults(ext_modelname=False)
    parser.set_defaults(policy_modelname=False)
    parser.set_defaults(gpus=[0])

    # if trained then require:
    parser.set_defaults(num_dl_workers=8)
    parser.set_defaults(batch_size=batch_size)
    parser.set_defaults(num_segments=num_segments)
    parser.set_defaults(fix_stride=5)
    parser.set_defaults(max_length=8)

    #parser.set_defaults(epochs=epochs)
    #parser.set_defaults(lr=ALPHA)
    #parser.set_defaults(weight_decay=0.0005)
    #parser.set_defaults(momentum=0.9)

    parser.set_defaults(gaussian_value=gaussian_value)

    parser.set_defaults(input_dtype=input_dtype)
    parser.set_defaults(optimizer="Adam")
    parser.set_defaults(trace_file="/home/mbc2004/datasets/BlockConstruction/traces6.npy") # traces5.npy
    parser.set_defaults(log_dir=log_dir)
    parser.set_defaults(home_dir="/home/mbc2004")

    parser.set_defaults(dense_sample=dense_sample)
    parser.set_defaults(dense_rate=dense_rate)

    args, _ = parser.parse_known_args()

    return Parameters(args)




def parse_model_args():
    parser = argparse.ArgumentParser(description='Generate IADs from input files')

    # model command line args
    parser.add_argument('app', help='the checkpoint file to use with the model', choices=['bi', 'bs'])

    # whether the model should use D-ITR-L or not
    parser.add_argument('--ditrl', dest='use_ditrl', action='store_true', help='flag denotes that D-ITR-L should be applied')
    parser.set_defaults(use_ditrl=False)
    parser.add_argument('--trim', dest='trim_model', action='store_true', help='flag denotes that Model should be trained on observations only, and should not be used to generate a policy')
    parser.set_defaults(trim_model=False)
    parser.add_argument('--bottleneck_size', type=int, default=128, help='if using D-ITR-L what bottleneck size.')

    # whether the model is being trained
    parser.add_argument('--model_dir', default="saved_models")
    parser.add_argument('--output_dir', default="csv_output")

    parser.add_argument('--save_id', default="", help='model_id to restore')
    parser.add_argument('--pretrain_modelname', default="/home/mbc2004/models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth", help='load the backbone model features from this file; these features can be fine-tuned and are not fixed')
    parser.add_argument('--backbone_modelname', default=False, help='load the backbone model features from this file; these features are fixed when this parameter is present')
    parser.add_argument('--ext_modelname', default=False, help='load the D-ITR-L model features from this file; these features are fixed when this parameter is present')
    parser.add_argument('--ditrl_modelname', default=False, help='load the D-ITR-L model features from this file; these features are fixed when this parameter is present')
    parser.add_argument('--policy_modelname', default=False, help='load the Policy model features from this file; these features are fixed when this parameter is present')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])

    # if trained then require:
    parser.add_argument('--num_dl_workers', type=int, default=8, help='the number of workers for the DataLoader')
    parser.add_argument('--batch_size', type=int, default=BATCH, help='the number of segments to split a clip into')
    parser.add_argument('--num_segments', type=int, default=SEGMENTS, help='the number of segments to split a clip into')
    parser.add_argument('--fix_stride', type=int, default=5, help='the number of segments to split a clip into')
    parser.add_argument('--max_length', type=int, default=8, help='the length of a clip')

    parser.add_argument('--epochs', type=int, default=EPOCHS, help='gpu to run on')
    parser.add_argument('--lr', type=float, default=ALPHA, help='gpu to run on')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='the length of a clip')
    parser.add_argument('--momentum', type=float, default=0.9, help='the length of a clip')
    parser.add_argument('--log_dir', default="analysis/fig", help='the checkpoint file to use with the model')

    parser.add_argument('--gaussian_value', default=1, type=int, help='the checkpoint file to use with the model')

    parser.add_argument('--input_dtype', default="video", choices=["video", "itr"], help='the checkpoint file to use with the model')
    parser.add_argument('--optimizer', default="Adam", choices=["Adam", "SGD"], help='the checkpoint file to use with the model')
    parser.add_argument('--trace_file', default="/home/mbc2004/datasets/BlockConstruction/traces.npy")

    return Parameters(parser.parse_args())


