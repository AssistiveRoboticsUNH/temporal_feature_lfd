import os

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

BASE_MODEL_DIR = "base_models"
MODEL_SAVE_DIR = "saved_models"

# input parameters
INPUT_FRAMES = 64  # 16

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
                     model_save_dir=MODEL_SAVE_DIR,
                     base_model_dir=BASE_MODEL_DIR,

                     input_frames=INPUT_FRAMES
                     ):
            self.gpus = gpus
            self.dataloader_workers = dataloader_workers

            self.batch_size = batch_size
            self.epochs = epochs  # number of epochs to run experiments for
            self.lr = lr  
            self.weight_decay = weight_decay # ?
            self.momentum = momentum  # ?

            self.gaussian_value = gaussian_value

            self.home_dir = home_dir

            self.base_model_dir = base_model_dir
            self.model_save_dir = model_save_dir

            self.input_frames = input_frames

            self.model = "unassigned"
            self.application = "unassigned"

        class ApplicationDef:
            def __init__(self, app):
                self.app = app
                if app == "block_construction":
                    self.file_directory = "/home/mbc2004/datasets/BlockConstruction"
                    self.trace_file = os.path.join(self.file_directory, "traces6.npy")
                    self.obs_label_list = {"n": 0, "r": 1, "rr": 2, "rrr": 3, "g": 4, "gb": 5, "bg": 6, "b": 7}
                    self.act_label_list = {"N": 0, "R": 1, "G": 2, "B": 3}

                    #models
                    self.tsm = {"filename": "c_backbone_tsm_1_bn16", "bottleneck": 16}
                    self.wrn = {"filename": "c_backbone_wrn_2_bn16", "bottleneck": 16}#c_backbone_wrn_1_bn64
                    self.i3d = {"filename": "c_backbone_i3d_1_bn8", "bottleneck": 8}
                    self.vgg = {"filename": "c_backbone_vgg_2_bn32", "bottleneck": 32}

                elif app == "block_construction_timed":
                    self.file_directory = "/home/mbc2004/datasets/BlockConstructionTimed"
                    self.trace_file = os.path.join(self.file_directory, "traces6.npy")
                    self.obs_label_list = {"n": 0, "r": 1, "rr": 2, "rrr": 3, "g": 4, "gb": 5, "bg": 6, "b": 7}
                    #self.trace_file = os.path.join(self.file_directory, "traces_rgb.npy")
                    #self.obs_label_list = {"n": 0, "r": 1, "g": 2, "b": 3}
                    self.act_label_list = {"N": 0, "R": 1, "G": 2, "B": 3}

                    # models
                    self.tsm = {"filename": "c_backbone_tsm_1_bn16", "bottleneck": 16}
                    self.wrn = {"filename": "c_backbone_wrn_0_bn16", "bottleneck": 16} #c_backbone_wrn_4_bn64/c_backbone_wrn_0_bn32"
                    self.i3d = {"filename": "c_backbone_i3d_1_bn16", "bottleneck": 16}
                    self.vgg = {"filename": "c_backbone_vgg_0_bn32", "bottleneck": 32}

                elif app == "tea_making":
                    self.file_directory = "/home/mbc2004/datasets/TeaMaking2"
                    self.obs_label_list = {"add_milk": 0, "add_sugar": 1, "add_tea_bag": 2, "add_water": 3,
                                           "nothing": 4, "stir": 5, "toggle_on_off": 6}
                    self.act_label_list = None  # {"N": 0, "R": 1, "G": 2, "B": 3}

                    # models
                    self.tsm = {"filename": "", "bottleneck":0}
                    self.wrn = {"filename": "", "bottleneck":0}
                    self.i3d = {"filename": "c_backbone_i3d_0", "bottleneck": 16}
                    self.vgg = {"filename": "c_backbone_vgg_1", "bottleneck":32}

                elif app == "ikea":
                    self.file_directory = "/home/mbc2004/datasets/IKEA"
                    label_path = os.path.join(*[self.file_directory, "frames",  "train"])
                    self.obs_label_list = {k: v for v, k in enumerate(os.listdir(label_path))}
                    self.act_label_list = None  # {"N": 0, "R": 1, "G": 2, "B": 3}

                    # models
                    self.tsm = {"filename": "", "bottleneck": 0}
                    self.wrn = {"filename": "", "bottleneck": 0}
                    self.i3d = {"filename": "", "bottleneck": 16}
                    self.vgg = {"filename": "c_backbone_vgg_0", "bottleneck": 32}

                elif app == "crepe":
                    self.file_directory = "/home/mbc2004/datasets/Crepe"
                    label_path = os.path.join(*[self.file_directory, "actions", "frames",  "train"])
                    self.obs_label_list = {k: v for v, k in enumerate(os.listdir(label_path))}
                    self.act_label_list = None  # {"N": 0, "R": 1, "G": 2, "B": 3}

                    # models
                    self.tsm = {"filename": "", "bottleneck": 0}
                    self.wrn = {"filename": "", "bottleneck": 0}
                    self.i3d = {"filename": "", "bottleneck": 16}
                    self.vgg = {"filename": "c_backbone_vgg_0", "bottleneck": 32}

                self.num_labels = len(self.obs_label_list)

            def print_application(self):
                print("application - "+self.app)
                print("\tdirectory - " + self.file_directory)
                #print("\ttrace_file - " + self.trace_file)

        def set_application(self, app):
            self.application = self.ApplicationDef(app)
            self.base_model_dir += '_' + app
            self.model_save_dir += '_' + app

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

            assert self.application != "unassigned", "ERROR: call the set_application function before the set_model_params function"

            # pretrain_model_name = None
            # save_id = "classifier_bottleneck_r21d0"

            if model_id == Backbone.TSM:
                from model.backbone_model.backbone_tsm import BackboneTSM as backbone_class
                pretrain_model_name = os.path.join(self.home_dir,
                    "models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth")
                # save_id = "classifier_bottleneck_tsm3"
                #save_id = "c_backbone_tsm_0"

                save_id = self.application.tsm["filename"]
                bottleneck = self.application.tsm["bottleneck"]

                self.model = self.ModelDef("tsm", bottleneck, [2048], [64], 7, backbone_class,
                                           pretrain_model_name=pretrain_model_name,
                                           save_id=save_id)

            elif model_id == Backbone.TRN:
                from model.backbone_model.backbone_trn import BackboneTRN as backbone_class
                pretrain_model_name = os.path.join(self.home_dir,
                    "models/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar")

                save_id = self.application.trn["filename"]
                bottleneck = self.application.trn["bottleneck"]

                self.model = self.ModelDef("trn", bottleneck, [2048], [64], 7, backbone_class,
                                           pretrain_model_name=pretrain_model_name)

            elif model_id == Backbone.WRN:
                from model.backbone_model.backbone_wrn import BackboneWideResNet as backbone_class
                #save_id = "classifier_bottleneck_wrn1"

                save_id = self.application.wrn["filename"]
                bottleneck = self.application.wrn["bottleneck"]

                self.model = self.ModelDef("wrn", bottleneck, [2048], [64], 7, backbone_class,
                                           save_id=save_id)

            elif model_id == Backbone.VGG:
                from model.backbone_model.backbone_vgg import BackboneVGG as backbone_class
                # save_id = "classifier_bottleneck_vgg0"  # BN 32
                #save_id = "c_backbone_vgg_1"  # BN 32?

                save_id = self.application.vgg["filename"]
                bottleneck = self.application.vgg["bottleneck"]

                self.model = self.ModelDef("vgg", bottleneck, [512], [64], 7, backbone_class,
                                           save_id=save_id)

            elif model_id == Backbone.I3D:
                original_size = [64, 192, 256, 832, 1024, 1024]
                iad_frames = [32, 32, 32, 16, 8, 8]

                from model.backbone_model.backbone_i3d import BackboneI3D as backbone_class
                pretrain_model_name = os.path.join(self.home_dir,
                    "models/rgb_imagenet.pt")

                save_id = self.application.i3d["filename"]
                bottleneck = self.application.i3d["bottleneck"]

                self.model = self.ModelDef("i3d", bottleneck, original_size, iad_frames, 7, backbone_class,
                                           pretrain_model_name=pretrain_model_name,
                                           save_id=save_id,
                                           end_point=end_point)

    return Params()
