"""
This code is for the training the pipeline section of D-ITR-L using the backbone model. This code also saves the frames
as ITRs
"""
import torch
from PIL import Image
import numpy as np
import os

from image_capture import read_file

def run(lfd_params, model):
    # Create DataLoaders
    train_loader = lfd_params.create_dataloader(lfd_params, "train", shuffle=False, verbose=True)
    eval_loader = lfd_params.create_dataloader(lfd_params, "evaluation", shuffle=False, verbose=True)

    # put model on GPU
    rgb_model = model.rgb_net
    net = torch.nn.DataParallel(rgb_model, device_ids=lfd_params.args.gpus).cuda()
    net.eval()

    # generate ITRs
    for data_loader in [train_loader, eval_loader]:
        for i, data_packet in enumerate(data_loader):

            obs, state, action, filename = data_packet

            # input shapes
            if i == 0:
                print("obs: ", obs.shape)
                print("state: ", state.shape)

            # compute output
            activation_map = net(obs)
            activation_map = activation_map.detach().cpu().numpy()

            for n, file in enumerate(filename):

                iad = model.pipeline.convert_activation_map_to_iad(activation_map)
                print("iad:", iad.shape)
                sparse_map = model.pipeline.convert_iad_to_sparse_map(iad)
                print("sparse_map:", len(sparse_map))

                rgb_image = read_file(file)
                print("rgb_image:", rgb_image.shape)


                # format new save name
                save_id = file.split('/')
                file_id = save_id[-1]+".npz"
                save_id = save_id[:save_id.index("frames")] + ["itrs"] + save_id[save_id.index("frames") + 1:-1]
                save_id = '/' + os.path.join(*save_id)

                # create a directory to save the ITRs in
                #if not os.path.exists(save_id):
                #    os.makedirs(save_id)

                save_id = os.path.join(save_id, file_id)

                print("n: {0}, filename: {1}, save_id: {2}".format(n, file, save_id))

            print("generate ITRs: iter: {:6d}/{:6d}".format(i, len(data_loader)))

    # save trained model parameters
    out_filename = lfd_params.generate_modelname()
    model.save_model()

    return out_filename


if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()

    from model.temporal_feature_extractor import TemporalFeatureExtractor
    model_obj = TemporalFeatureExtractor(lfd_params_obj, use_pipeline=True, train_pipeline=True, use_model=False,
                                         train_model=False)

    run(lfd_params_obj, model_obj)






















import argparse


def visualize_iad(model, filename, num_segments=8, image_tmpl='image_{:05d}.jpg', output_filename="image_stitch.png"):

    total_num_frames = len(os.listdir(filename))
    print("total num frames:", total_num_frames)
    print("num segments:", num_segments)

    # collect frames
    images = []
    idxs = np.linspace(0, max(1, total_num_frames-1), num=num_segments, dtype=int)+1

    for idx in idxs:
        images.append(Image.open(os.path.join(filename, image_tmpl.format(idx))).convert('RGB'))

    # stitch frames together
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    model.rgb_net()



    img = images[0]
    for i in range(1, len(images)):
        img = get_concat_h(img, images[i])

    # save to file
    img.save(os.path.join(args.fig_dir, output_filename))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate IADs from input files')
    parser.add_argument('input_file', help='the checkpoint file to use with the model')
    parser.add_argument('--fig_dir', default="analysis/fig",help='the checkpoint file to use with the model')
    parser.add_argument('--num_segments', default=8, type=int,help='the checkpoint file to use with the model')
    parser.add_argument('--mode', default="train", choices=["train", "eval"],
                        help='the checkpoint file to use with the model')

    args = parser.parse_args()

    outname = args.input_file.split("/")[-1]
    print("outname:", outname)

    num_segments=16

    from parameter_parser import default_model_args
    lfd_params_obj = default_model_args(trim=True,
                                        use_ditrl=True,
                                        save_id="model0_train",
                                        backbone_model="saved_models/saved_model_model0.backbone.pt",
                                        num_segments=num_segments)

    from model.temporal_feature_extractor import TemporalFeatureExtractor
    model_obj = TemporalFeatureExtractor(lfd_params_obj, use_pipeline=True, train_pipeline=False, use_model=False,
                                         train_model=False)

    visualize_iad(model_obj, args.input_file, output_filename=outname, num_segments=num_segments)

    # read_file(args, args.input_file, mode="train", image_tmpl='image_{:05d}.jpg', output_filename="image_train_"+outname+".png")
    # read_file(args, args.input_file, mode="eval", image_tmpl='image_{:05d}.jpg', output_filename="image_eval_"+outname+".png")