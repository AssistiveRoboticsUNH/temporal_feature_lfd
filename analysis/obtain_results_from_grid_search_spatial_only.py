import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


if __name__ == "__main__":
    target = "gaussian_value"

    parser = argparse.ArgumentParser(description='Generate IADs from input files')
    parser.add_argument('spatial_results_file', help='the checkpoint file to use with the model')
    parser.add_argument('--output_file', default="gaussian_accuracy.png", help='the checkpoint file to use with the model')
    args = parser.parse_args()

    # load spatial data
    spatial_df = pd.read_csv(args.spatial_results_file)
    spatial_df["model"] = ["spatial"] * len(spatial_df)
    spatial_df["correct"] = spatial_df["expected_action"] == spatial_df["observed_action"]
    spatial_df = spatial_df.groupby([target, "repeat"]).mean().reset_index()
    spatial_df_mean = spatial_df.groupby([target]).mean().reset_index()
    spatial_df_std = spatial_df.groupby([target]).std().reset_index()

    # df = spatial_df.append(temporal_df)
    labels = spatial_df[target].to_numpy()
    spatial_data = spatial_df["correct"].to_numpy()
    df = pd.DataFrame({"spatial": spatial_df_mean, 'std': spatial_df_std}, index=labels)

    df.plot.bar()
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel(target)

    plt.errorbar(df.index, df['spatial'], yerr=df['std'])

    plt.tight_layout()

    # save plt
    #plt.show()
    plt.savefig(os.path.join("analysis/fig/", args.output_file))



