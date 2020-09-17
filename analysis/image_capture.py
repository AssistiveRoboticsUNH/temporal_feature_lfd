import os
from PIL import Image, ImageFilter
import numpy as np 

import argparse

# need to remove ros path before I can import cv2
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2


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


def applyGaussian(img_array, gaussian_value):
    image_out = []
    for img in img_array:
        image_out.append(img.filter(ImageFilter.GaussianBlur(gaussian_value)))
    return image_out


def applyDifferenceMask(img_array, gaussian_value=1, noise_kernel=3):
    # convert to gray scale
    image_out = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (noise_kernel, noise_kernel))
    backSub = cv2.createBackgroundSubtractorMOG2()

    # apply background removal to first frame of video
    frame = np.array(img_array[0].filter(ImageFilter.GaussianBlur(gaussian_value)))
    _ = backSub.apply(frame)

    for img in img_array[1:]:

        frame = np.array(img.filter(ImageFilter.GaussianBlur(gaussian_value)))  # smooth video to reduce slight variances

        fg_mask = backSub.apply(frame)  # remove background
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        #print(fg_mask, np.min(fg_mask), np.max(fg_mask))
        fg_mask[fg_mask > 0] =  255
        #fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel) # filter noise in mask
        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0  # convert to float values

        img_f = (fg_mask * frame).astype(np.uint8)  # apply mask to frame
        img_f = Image.fromarray(img_f)  # convert frame to PIL image
        image_out.append(img_f)

    image_out.append(Image.fromarray(np.zeros_like(img_f)))  # pad data to required length

    return image_out


def read_file(num_segments, input_file, mode="train", image_tmpl='image_{:05d}.jpg', output_filename="image_stitch.png",
              save_file=True, merge_images=True, gaussian_value=1, kernel_size=3):

    total_num_frames = len(os.listdir(input_file))
    print("total num frames:", total_num_frames)
    print("num segments:", num_segments)
    print("gaussian value:", gaussian_value)

    # collect frames
    images = []
    if mode == "train":
        idxs = np.linspace(0, max(1, total_num_frames-1), num=num_segments, dtype=int)+1
    else:
        idxs = np.linspace(0, max(1, total_num_frames-1), num=num_segments*10, dtype=int)+1

    for idx in idxs:
        images.append(Image.open(os.path.join(input_file, image_tmpl.format(idx))).convert('RGB'))

    #images = applyGaussian(images, gaussian_value)
    #images = applyOpticalFlowMasking(images)
    images = applyDifferenceMask(images, gaussian_value, kernel_size)
    #images = applySaliencyMap(images)

    # stitch frames together
    if merge_images:
        img = images[0]
        for i in range(1, len(images)):
            img = get_concat_h(img, images[i])
    else:
        img = images

    # save to file
    if save_file:
        try:

            img.save(output_filename)
            print("file saved to: " + output_filename)
        except():
            print("ERROR: failed to save!")
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate IADs from input files')
    parser.add_argument('input_file', help='the checkpoint file to use with the model')
    parser.add_argument('--fig_dir', default="analysis/fig",help='the checkpoint file to use with the model')
    parser.add_argument('--num_segments', default=8, type=int,help='the checkpoint file to use with the model')
    parser.add_argument('--mode', default="train", choices=["train", "eval"],
                        help='the checkpoint file to use with the model')
    parser.add_argument('--gaussian_value', default=0, type=int, help='the checkpoint file to use with the model')

    args = parser.parse_args()

    num_segments = args.num_segments
    input_file = args.input_file

    outname = args.input_file.split("/")[-1]
    out_filepath = os.path.join(args.fig_dir, "image_train_"+outname+".png")
    print("out_filepath:", out_filepath)

    """
    read_file(num_segments, input_file, mode="train", image_tmpl='image_{:05d}.jpg',
              output_filename=out_filepath, gaussian_value=args.gaussian_value, kernel_size=(3,3))
    """
    img_stack = []
    for gv in range(4):
        for ks in range(3, 10, 2):
            img = read_file(num_segments, input_file, mode="train", image_tmpl='image_{:05d}.jpg',
                            save_file=False, gaussian_value=gv, kernel_size=ks)
            img_stack.append(img)

    img_s = img_stack[0]
    for img in img_stack[1:]:
        img_s = get_concat_v(img_s, img)

    img_s.show()
