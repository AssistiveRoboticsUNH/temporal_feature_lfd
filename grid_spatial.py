"""
This code is for the training of a network using only the backbone model
"""
from train_spatial import train
from eval_spatial import evaluate

from parameter_parser import parse_model_args
from model.spatial_feature_extractor import SpatialFeatureExtractor

import pandas as pd
import os

if __name__ == '__main__':

    lfd_params = parse_model_args()
    output_df = pd.DataFrame()

    for bottleneck_size in [128, 64, 32, 16, 8, 4]:

        # parameter changes
        lfd_params.args.bottleneck = bottleneck_size
        lfd_params.args.save_id = "grid_bottleneck_"+str(bottleneck_size)
        lfd_params.locate_model_files_from_save_id()

        """"""
        # train model
        print(" --- ")
        print("Begin Training with bottleneck: {0}".format(bottleneck_size))
        print(" --- ")
        model_obj = SpatialFeatureExtractor(lfd_params, is_training=True)
        train(lfd_params, model_obj, debug=False)
        print(" --- ")
        print("Finished Training with bottleneck: {0}".format(bottleneck_size))
        print(" --- ")
        """"""

        # evaluate model
        print(" --- ")
        print("Begin Evaluating with bottleneck: {0}".format(bottleneck_size))
        print(" === ")
        lfd_params.locate_model_files_from_save_id()
        model_obj = SpatialFeatureExtractor(lfd_params, is_training=False)
        df = evaluate(lfd_params, model_obj, debug=False)
        output_df = pd.concat([output_df, df])
        print(" --- ")
        print("Finished Evaluating with bottleneck: {0}".format(bottleneck_size))
        print(" === ")

    # analyze output of spatial
    out_filename = os.path.join(lfd_params.args.output_dir, "output_grid_spatial.csv")
    df.to_csv(out_filename)
