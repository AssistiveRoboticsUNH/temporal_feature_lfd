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

    #for bottleneck_size in [128, 64, 32, 16, 8, 4]:
    for gaussian_value in [0, 1, 2, 3, 4]:

        # parameter changes
        lfd_params.args.bottleneck = 4#bottleneck_size
        lfd_params.args.gaussian_value = gaussian_value
        lfd_params.args.save_id = "grid_gauss_"+str(gaussian_value)
        lfd_params.locate_model_files_from_save_id()


        # train model
        print(" --- ")
        print("Begin Training with value: {0}".format(gaussian_value))
        print(" --- ")
        model_obj = SpatialFeatureExtractor(lfd_params, is_training=True)
        train(lfd_params, model_obj, debug=False)
        print(" --- ")
        print("Finished Training with value: {0}".format(gaussian_value))
        print(" --- ")


        # evaluate model
        print(" --- ")
        print("Begin Evaluating with value: {0}".format(gaussian_value))
        print(" === ")
        lfd_params.locate_model_files_from_save_id()
        model_obj = SpatialFeatureExtractor(lfd_params, is_training=False)
        df = evaluate(lfd_params, model_obj, debug=False)
        df["gaussian_value"] = np.array([gaussian_value]*len(df))
        output_df = output_df.append(df)
        print(" --- ")
        print("Finished Evaluating with value: {0}".format(gaussian_value))
        print(" === ")

    # analyze output of spatial
    out_filename = os.path.join(lfd_params.args.output_dir, "output_grid_spatial.csv")
    output_df.to_csv(out_filename)
