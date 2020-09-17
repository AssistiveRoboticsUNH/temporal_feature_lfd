"""
This code is for the training of a network using only the backbone model
"""
from train_spatial import train
from eval_spatial import evaluate

from parameter_parser import parse_model_args
from model.spatial_feature_extractor import SpatialFeatureExtractor

import numpy as np
import os
import pandas as pd

if __name__ == '__main__':

    lfd_params = parse_model_args()
    output_df = pd.DataFrame()
    num_repeats = 5

    value_name = "bottleneck"
    #for bottleneck_size in [128, 64, 32, 16, 8, 4]:
    #for gaussian_value in [0, 1, 2, 3, 4]:
    for value in [4]:#[128, 64, 32, 16, 8, 4]:
        for r in range(num_repeats):

            # parameter changes
            lfd_params.args.bottleneck = value
            #lfd_params.args.gaussian_value = value
            lfd_params.args.save_id = "grid_"+value_name+"_"+str(value)+"_"+str(r)
            lfd_params.locate_model_files_from_save_id()

            # evaluate model
            print(" --- ")
            print("Begin Evaluating with value: {0}".format(value))
            print(" === ")
            lfd_params.locate_model_files_from_save_id()
            model_obj = SpatialFeatureExtractor(lfd_params, is_training=False)
            df = evaluate(lfd_params, model_obj, debug=False)

            df[value_name] = np.array([value]*len(df))
            df["repeat"] = np.array([r]*len(df))
            output_df = output_df.append(df)
            print(" --- ")
            print("Finished Evaluating with value: {0}".format(value))
            print(" === ")

    # analyze output of spatial
    out_filename = os.path.join(lfd_params.args.output_dir, "check_spatial.csv")
    output_df.to_csv(out_filename)

















"""
if __name__ == '__main__':

    lfd_params = parse_model_args()
    output_df = pd.DataFrame()
    num_repeats = 5

    value_name = "bottleneck"
    #for bottleneck_size in [128, 64, 32, 16, 8, 4]:
    #for gaussian_value in [0, 1, 2, 3, 4]:
    for value in [128, 64, 32, 16, 8, 4]:
        for r in range(num_repeats):

            # parameter changes
            lfd_params.args.bottleneck = value
            #lfd_params.args.gaussian_value = value
            lfd_params.args.save_id = "grid_"+value_name+"_"+str(value)+"_"+str(r)
            lfd_params.locate_model_files_from_save_id()

            # train model
            print(" --- ")
            print("Begin Training with value: {0}".format(value))
            print(" --- ")
            model_obj = SpatialFeatureExtractor(lfd_params, is_training=True)
            train(lfd_params, model_obj, debug=False)
            print(" --- ")
            print("Finished Training with value: {0}".format(value))
            print(" --- ")

            # evaluate model
            print(" --- ")
            print("Begin Evaluating with value: {0}".format(value))
            print(" === ")
            lfd_params.locate_model_files_from_save_id()
            model_obj = SpatialFeatureExtractor(lfd_params, is_training=False)
            df = evaluate(lfd_params, model_obj, debug=False)

            df[value_name] = np.array([value]*len(df))
            df["repeat"] = np.array([r]*len(df))
            output_df = output_df.append(df)
            print(" --- ")
            print("Finished Evaluating with value: {0}".format(value))
            print(" === ")

    # analyze output of spatial
    out_filename = os.path.join(lfd_params.args.output_dir, "output_grid_spatial_saliency_tanh_bottleneck.csv")
    output_df.to_csv(out_filename)
"""