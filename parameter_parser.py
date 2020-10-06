import argparse
import os
import datetime

ROOT_DIR = "/home/mbc2004/"
SEGMENTS = 16
EPOCHS = 200
ALPHA = 0.0005
BATCH = 3


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

        if self.args.save_id != "":
            self.locate_model_files_from_save_id()
        else:
            self.generate_save_id()

        if self.debug:
            print(self.args)

    def use_itrs(self, value=True):

        if value:
            if self.args.app == "bi":
                from datasets.social_greeting_dl import create_dataloader_itr
            elif self.args.app == "bs":
                from datasets.block_construction_dl import create_dataloader_itr
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

        from datasets.social_greeting_dl import create_dataloader
        self.create_dataloader = create_dataloader

    def setup_block_stacking(self):
        print("Loading block_construction_dl")
        self.file_directory = os.path.join(ROOT_DIR, "datasets/BlockConstruction/frames/")
        self.num_actions = 3 # 7
        self.num_hidden_state_params = 1

        self.use_aud = False
        self.checkpoint_file = os.path.join(ROOT_DIR, "models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth")
        self.trained_checkpoint_file = os.path.join(ROOT_DIR, "models/block_construction_tsm.pth")

        from datasets.block_construction_dl import create_dataloader
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


def default_model_args(use_ditrl=False,
                       trim_model=False,
                       save_id="",
                       backbone_model="/home/mbc2004/models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth",
                       num_segments=SEGMENTS,
                       gaussian_value=0):
    parser = argparse.ArgumentParser(description='Generate IADs from input files')

    # model command line args
    parser.set_defaults(app='bs')

    # whether the model should use D-ITR-L or not
    parser.set_defaults(use_ditrl=use_ditrl)
    parser.set_defaults(trim_model=trim_model)

    parser.set_defaults(bottleneck_size=128)

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
    parser.set_defaults(batch_size=BATCH)
    parser.set_defaults(num_segments=num_segments)
    parser.set_defaults(fix_stride=5)
    parser.set_defaults(max_length=8)

    parser.set_defaults(epochs=EPOCHS)
    parser.set_defaults(lr=ALPHA)
    parser.set_defaults(weight_decay=0.0005)
    parser.set_defaults(momentum=0.9)

    parser.set_defaults(gaussian_value=gaussian_value)


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

    return Parameters(parser.parse_args())


