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


def applyOpticalFlowMasking(img_array):
    # convert to gray scale
    image_out = []

    prev_gray = cv2.cvtColor(np.array(img_array[0]), cv2.COLOR_BGR2GRAY)
    for img in img_array[1:]:
        # get optical flow
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 7, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        prev_gray = gray

        src = np.array(img)
        # magnitude *= 5
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mask = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR) / 255
        print("mask:", np.max(mask), np.min(mask))
        # mask image
        img_out = Image.fromarray((src * mask).astype(np.uint8))

        image_out.append(img_out)
    # add an additional image to maintain segment length
    image_out.append(img)
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
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # filter noise in mask
        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0  # convert to float values

        img_f = (fg_mask * frame).astype(np.uint8)  # apply mask to frame
        img_f = Image.fromarray(img_f)  # convert frame to PIL image
        image_out.append(img_f)

    image_out.append(Image.fromarray(np.zeros_like(img_f)))  # pad data to required length

    return image_out


def applySaliencyMap(img_array):


    saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
    saliency.setImagesize(img_array[0].width, img_array[0].height)
    saliency.init()

    #saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    #saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    #saliency_map = np.zeros_like(np.array(img_array[0])).astype(np.float32)
    gray = cv2.cvtColor(np.array(img_array[0]), cv2.COLOR_BGR2GRAY)
    for i in range(52):
        _ = saliency.computeSaliency(gray)

    image_out = []
    for img in img_array:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)#.astype(np.float32) / 255
        print("0:", gray, gray.dtype)
        (success, saliency_map) = saliency.computeSaliency(gray)#, saliency_map)
        print("success:", success, saliency_map.dtype)
        print("1:", saliency_map)
        saliency_map = (saliency_map * 255).astype("uint8")
        print("2:", saliency_map)
        img_f = Image.fromarray(saliency_map)

        image_out.append(img_f)
    return image_out




def read_file(num_segments, input_file, mode="train", image_tmpl='image_{:05d}.jpg', output_filename="image_stitch.png",
              save_file=True, merge_images=True, gaussian_value=1):

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
    images = applyDifferenceMask(images)
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
            filepath = os.path.join(args.fig_dir, output_filename)
            img.save(filepath)
            print("file saved to: " + filepath)
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

    outname = args.input_file.split("/")[-1]
    print("outname:", outname)

    num_segments = args.num_segments
    input_file = args.input_file

    read_file(num_segments, input_file, mode="train", image_tmpl='image_{:05d}.jpg',
              output_filename="image_train_"+outname+".png", gaussian_value=args.gaussian_value)
    #read_file(args, args.input_file, mode="eval", image_tmpl='image_{:05d}.jpg',
    #         output_filename="image_eval_"+outname+".png")