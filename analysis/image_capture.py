import sys, os
from PIL import Image
import numpy as np 

import argparse


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


def read_file(num_segments, input_file, mode="train", image_tmpl='image_{:05d}.jpg', output_filename="image_stitch.png",
              save_file=True, merge_images=True):

    total_num_frames = len(os.listdir(input_file))
    print("total num frames:", total_num_frames)
    print("num segments:", num_segments)

    # collect frames
    images = []
    if mode == "train":
        idxs = np.linspace(0, max(1, total_num_frames-1), num=num_segments, dtype=int)+1
    else:
        idxs = np.linspace(0, max(1, total_num_frames-1), num=num_segments*10, dtype=int)+1

    for idx in idxs:
        images.append(Image.open(os.path.join(input_file, image_tmpl.format(idx))).convert('RGB'))

    # stitch frames together
    if merge_images:
        img = images[0]
        for i in range(1, len(images)):
            img = get_concat_h(img, images[i])

    # save to file
    if save_file:
        img.save(os.path.join(args.fig_dir, output_filename))
    return img


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

    num_segments = args.num_segments
    input_file = args.input_file

    read_file(num_segments, input_file, mode="train", image_tmpl='image_{:05d}.jpg', output_filename="image_train_"+outname+".png")
    #read_file(args, args.input_file, mode="eval", image_tmpl='image_{:05d}.jpg', output_filename="image_eval_"+outname+".png")