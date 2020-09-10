"""
This code is for the training of a network using only the backbone model
"""
from train_temporal_pipeline import train_pipeline
from train_temporal_ext import train_model
from eval_temporal import evaluate

from parameter_parser import parse_model_args
from model.temporal_feature_extractor import TemporalFeatureExtractor

import numpy as np
import os
import pandas as pd


if __name__ == '__main__':

    lfd_params = parse_model_args()
    output_df = pd.DataFrame()

    for bottleneck_size in [128, 64, 32, 16, 8, 4]:

        # parameter changes
        lfd_params.args.bottleneck = bottleneck_size
        lfd_params.args.save_id = "grid_bottleneck_"+str(bottleneck_size)
        lfd_params.locate_model_files_from_save_id()
        lfd_params.use_itrs(False)

        # train model
        print(" --- ")
        print("Begin Training with bottleneck: {0}".format(bottleneck_size))
        print(" --- ")
        # train pipeline
        model_obj = TemporalFeatureExtractor(lfd_params, use_pipeline=True, train_pipeline=True, use_model=False,
                                             train_model=False)
        train_pipeline(lfd_params, model_obj, debug=False)

        # train model
        lfd_params.use_itrs(True)
        model_obj = TemporalFeatureExtractor(lfd_params, use_pipeline=False, train_pipeline=False, use_model=True,
                                             train_model=True)
        train_model(lfd_params, model_obj, debug=False)
        print(" --- ")
        print("Finished Training with bottleneck: {0}".format(bottleneck_size))
        print(" --- ")

        # evaluate model
        print(" --- ")
        print("Begin Evaluating with bottleneck: {0}".format(bottleneck_size))
        print(" === ")
        lfd_params.locate_model_files_from_save_id()
        model_obj = TemporalFeatureExtractor(lfd_params, use_pipeline=False, train_pipeline=False, use_model=True,
                                             train_model=False)
        df = evaluate(lfd_params, model_obj, debug=False)
        df["bottleneck"] = np.array([bottleneck_size] * len(df))
        output_df = output_df.append(df)
        print(" --- ")
        print("Finished Evaluating with bottleneck: {0}".format(bottleneck_size))
        print(" === ")

    # analyze output of spatial
    out_filename = os.path.join(lfd_params.args.output_dir, "output_grid_temporal.csv")
    output_df.to_csv(out_filename)
