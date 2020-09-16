import os
import argparse
from image_capture import read_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate IADs from input files')
    parser.add_argument('input_file', help='the checkpoint file to use with the model')
    parser.add_argument('--fig_dir', default="analysis/fig", help='the checkpoint file to use with the model')
    parser.add_argument('--num_segments', default=16, type=int, help='the checkpoint file to use with the model')
    parser.add_argument('--mode', default="train", choices=["train", "eval"],
                        help='the checkpoint file to use with the model')

    args = parser.parse_args()

    for observations in os.listdir(args.input_file):
        for file in os.listdir(os.path.join(args.input_file, observations)):
            input_filename = os.path.join(args.input_file, observations, file)
            print("input_filename: ", input_filename)

            read_file(args.num_segments, input_filename, mode="train", image_tmpl='image_{:05d}.jpg',
                      output_filename="image_train_"+file+".png")
