import os
import pandas as pd
import numpy as np

from enums import suffix_dict, model_dict, Suffix
from parameter_parser import default_model_params

#from run_classification import generate_iad_files as generate_iad_files_code
from execute import make_model_name, define_model
from datasets.utils import create_dataloader

import torch
import torchvision
from model.backbone_model.tsm.ops.transforms import *

'''
ssv2_filenames = {
    "train": os.path.join(*[dataset_loc, "annotation", "something-something-v2-train.json"]),
    "validation": os.path.join(*[dataset_loc, "annotation", "something-something-v2-validation.json"]),
    "test": os.path.join(*[dataset_loc, "annotation", "something-something-v2-test.json"]),
    "labels": os.path.join(*[dataset_loc, "annotation", "something-something-v2-labels.json"])
}
'''

class Dataset20BN:
    def __init__(self, dataset_loc, id_name):
        self.dataset_loc = dataset_loc
        self.id_name = id_name
        self.files_names = {
            "train": os.path.join(*[dataset_loc, "annotation", self.id_name + "-train.csv"]),
            "validation": os.path.join(*[dataset_loc, "annotation", self.id_name + "-validation.csv"]),
            "test": os.path.join(*[dataset_loc, "annotation", self.id_name + "-test.csv"]),
            "labels": os.path.join(*[dataset_loc, "annotation", self.id_name + "-labels.csv"])
        }

        label_file = open('r', self.files_names["labels"]).read().split(',')
        self.labels_dict = {k: v for k, v in enumerate(label_file)}

        self.data = []
        for file_list in [self.files_names["train"], self.files_names["validation"]]:
            ifile = open('r', file_list)
            line = ifile.readline()
            while len(line) != 0:
                filename, label = line[:-1].split(';')
                self.data.append((filename, self.labels_dict[label]))
                line = ifile.readline()

        self.transform = torchvision.transforms.Compose([
                GroupScale(224),
                GroupCenterCrop(224),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                IdentityTransform(),
                ])

    def parse_obs(self, filename):
        imgs = [Image.open(x) for x in os.listdir(filename)]
        return self.transform(imgs)

    def __index__(self, idx):
        filename, label = self.data[idx]
        video = self.parse_obs(filename)
        return video, label, filename

    def __len__(self):
        return len(self.data)


def generate_iad_files(args, lfd_params, model, verbose=True):
    backbone_id = args.model

    from datasets.dataset_video import DatasetVideo as CustomDataset
    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, "evaluation", verbose=True,
                            num_segments=lfd_params.input_frames)
    data_loader = create_dataloader(dataset, lfd_params, "evaluation", shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet

        # compute output
        iad = net(obs)
        iad = iad.detach().cpu().numpy()

        for n, file in enumerate(filename):

            # format new save name
            save_id = file.split('/')
            file_id = save_id[-1] + ".npz"
            save_id = save_id[:save_id.index("frames")] + ["iad_" + args.model] + save_id[save_id.index("frames") + 1:-1]
            save_id = '/' + os.path.join(*save_id)

            # create a directory to save the ITRs in
            #if not os.path.exists(save_id):
            #    os.makedirs(save_id)

            save_id = os.path.join(save_id, file_id)

            if verbose:
                print("n: {0}, filename: {1}, saved_id: {2}".format(n, file, save_id))

            # save ITR to file with given name
            print(save_id)
            print("iad.shape:", iad[n].shape)


            #np.savez(save_id, data=iad[n], label=label)


def generate_files(args, lfd_params):
    print("Generate Files...")

    print("Generate IAD...")
    model = define_model(args, lfd_params, train=False, app='c', suffix=Suffix.GENERATE_IAD,
                         use_bottleneck=False,
                         backbone=True)
    generate_iad_files(args, lfd_params, model)

    print("Done!")


def execute_func(args, lfd_params, backbone=False):
    # generate files
    generate_files(args, lfd_params)


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    #parser.add_argument('app', help='classifier(c)/policy_learner(pl)', choices=['c', 'pl'])
    parser.add_argument('model', help='model_id', choices=model_dict.keys())
    #arser.add_argument('suffix', help='suffix', choices=['backbone', 'linear', 'lstm', 'tcn', 'ditrl'])

    #parser.set_defaults(generate_files=False)
    #parser.add_argument('--gen', help='generate_files', dest='generate_files', action='store_true')
    #parser.set_defaults(eval_only=False)
    #parser.add_argument('--eval', help='evaluate only', dest='eval_only', action='store_true')

    #parser.add_argument('--frames', help='number of frames', default=64, type=int)
    #parser.add_argument('--repeat', help='repeat code runs', default=1, type=int)
    parser.add_argument('--application', help='application', default="jester",
                        choices=['jester', 'ssv2'])

    return parser.parse_args()


########

if __name__ == '__main__':
    args = parse_exec_args()
    lfd_params = default_model_params()
    lfd_params.set_application(args.application)
    lfd_params.set_model_params(model_dict[args.model], end_point=-1)
    #lfd_params.input_frames = args.frames

    execute_func(args, lfd_params)
