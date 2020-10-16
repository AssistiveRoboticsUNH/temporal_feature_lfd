"""
This code is for the training of a network using only the backbone model
"""
from old_executables.train_spatial import train
from old_executables.eval_spatial import evaluate

from parameter_parser import parse_model_args
from model.spatial_feature_extractor import SpatialFeatureExtractor

import numpy as np
import os
import pandas as pd


if __name__ == '__main__':

    lfd_params = parse_model_args()
    output_df = pd.DataFrame()
    num_repeats = 5   # 5

    value_name = "bottleneck"
    for value in [64]:
        for r in range(num_repeats):

            # parameter changes
            lfd_params.args.bottleneck_size = value
            lfd_params.args.lr = 0.0001
            lfd_params.clear_model_files_from_save_id()
            lfd_params.args.save_id = "tsm_"+value_name+"_"+str(value)+"_"+str(r)
            lfd_params.locate_model_files_from_save_id()

            lfd_params.print_params()

            # train model
            print(" --- ")
            print("Begin Training with value: {0}, repeat:{1}".format(value, r))
            print(" --- ")
            model_obj = SpatialFeatureExtractor(lfd_params, is_training=True)
            train(lfd_params, model_obj, debug=False)
            print(" --- ")
            print("Finished Training with value: {0}, repeat:{1}".format(value, r))
            print(" --- ")

            # evaluate model
            print(" --- ")
            print("Begin Evaluating with value: {0}, repeat:{1}".format(value, r))
            print(" === ")
            lfd_params.locate_model_files_from_save_id()
            model_obj = SpatialFeatureExtractor(lfd_params, is_training=False)

            # train dataset
            df = evaluate(lfd_params, model_obj, mode="train", debug=False)
            df["mode"] = np.array(["train"] * len(df))
            df[value_name] = np.array([value] * len(df))
            df["repeat"] = np.array([r] * len(df))
            output_df = output_df.append(df)

            # test dataset
            df = evaluate(lfd_params, model_obj, mode="evaluation", debug=False)
            df["mode"] = np.array(["evaluation"] * len(df))
            df[value_name] = np.array([value]*len(df))
            df["repeat"] = np.array([r]*len(df))
            output_df = output_df.append(df)
            print(" --- ")
            print("Finished Evaluating with value: {0}, repeat:{1}".format(value, r))
            print(" === ")

    # analyze output of spatial
    out_filename = os.path.join(lfd_params.args.output_dir, "tsm_"+value_name+"_"+str(value)+".csv")
    output_df.to_csv(out_filename)