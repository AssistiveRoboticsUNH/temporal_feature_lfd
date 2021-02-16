import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def accuracy(spatial_df, temporal_df, target, title="", output_filename=""):
    def get_mean_and_std(df):

        df["correct"] = df["expected_action"] == df["observed_action"]
        df = df.groupby([target, "repeat"]).mean().reset_index()
        df_mean = df.groupby([target]).mean().reset_index()
        df_std = df.groupby([target]).std().reset_index()
        data_mean = df_mean["correct"].to_numpy()
        data_std = df_std["correct"].to_numpy()

        return data_mean, data_std

    spatial_data_mean, spatial_data_std = get_mean_and_std(spatial_df)
    temporal_data_mean, temporal_data_std = get_mean_and_std(temporal_df)

    labels = spatial_df[target].unique()

    #print(labels)
    #print(spatial_data_mean)
    #print(spatial_data_std)

    df = pd.DataFrame({"spatial": spatial_data_mean, "temporal": temporal_data_mean}, index=labels)

    print(df)

    df.plot.bar(yerr=spatial_data_std, ecolor='black', capsize=5)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel(target)

    plt.title(title)
    plt.tight_layout()

    # save plt
    if output_filename == "":
        plt.show()
    else:
        plt.savefig(output_filename)


def breakdown(df, target, title="", output_filename=""):
    spatial_df["model"] = ["spatial"] * len(spatial_df)
    spatial_df["correct"] = spatial_df["expected_action"] == spatial_df["observed_action"]
    spatial_df = spatial_df.groupby([target, "repeat", "expected_action"]).mean().reset_index()

    spatial_df_mean = spatial_df.groupby([target, "expected_action"]).mean().reset_index()
    # spatial_df_std = spatial_df.groupby([target, "expected_action"]).std()#.reset_index()

    label_order = [1,3,4,2,0,5,6]
    label_dict = {0: 'r', 1: 'g', 2: 'b', 3: 'gb', 4: 'bg', 5: 'rr', 6: 'rrr'}  # matches labels in block construction
    colors = {"r": "r", "g": "g", "b": "b", "bg": "cyan", "gb": "springgreen", "rr": "indianred", "rrr": "brown"}
    columns = {}
    for k in label_order:
        columns[label_dict[k]] = spatial_df_mean.loc[spatial_df_mean['expected_action'] == k]["correct"].to_numpy()

    labels = spatial_df[target].unique()
    df = pd.DataFrame(columns, index=labels)

    df.plot.bar(color=colors)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel(target)

    plt.title(title)
    plt.tight_layout()

    # save plt
    if output_filename == "":
        plt.show()
    else:
        plt.savefig(output_filename)

if __name__ == "__main__":
    #target_label = "saliency_tanh"
    #target_label = "gaussian_value"
    target_label = "bottleneck"

    parser = argparse.ArgumentParser(description='Generate IADs from input files')
    parser.add_argument('spatial_results_file', help='the checkpoint file to use with the model')
    parser.add_argument('temporal_results_file', help='the checkpoint file to use with the model')
    parser.add_argument('--output_file', default="", help='the checkpoint file to use with the model')
    args = parser.parse_args()

    output_filename_acc = "" if args.output_file == "" else os.path.join("analysis/fig/", args.output_file + "_accuracy.png")
    output_filename_bd = "" if args.output_file == "" else os.path.join("analysis/fig/", args.output_file + "_breakdown.png")

    # load spatial data
    spatial_df_src = pd.read_csv(args.spatial_results_file)
    temporal_df_src = pd.read_csv(args.spatial_results_file)

    # run analysis
    accuracy(spatial_df_src, temporal_df_src, target_label, title=target_label+" Accuracy", output_filename=output_filename_acc)
    breakdown(spatial_df_src, temporal_df_src, target_label, title=target_label+" Breakdown", output_filename=output_filename_bd)



"""
if __name__ == "__main__":
    target = "bottleneck"

    parser = argparse.ArgumentParser(description='Generate IADs from input files')
    parser.add_argument('spatial_results_file', help='the checkpoint file to use with the model')
    parser.add_argument('temporal_results_file', default="analysis/fig", help='the checkpoint file to use with the model')
    parser.add_argument('--output_file', default="accuracy.png", help='the checkpoint file to use with the model')
    args = parser.parse_args()

    # load spatial data
    spatial_df = pd.read_csv(args.spatial_results_file)
    spatial_df["model"] = ["spatial"] * len(spatial_df)
    spatial_df["correct"] = spatial_df["expected_action"] == spatial_df["observed_action"]
    spatial_df = spatial_df.groupby([target, "repeat"]).mean().reset_index()

    # load temporal data
    temporal_df = pd.read_csv(args.temporal_results_file)
    temporal_df["model"] = ["temporal"] * len(temporal_df)
    temporal_df["correct"] = temporal_df["expected_action"] == temporal_df["observed_action"]
    temporal_df = temporal_df.groupby([target, "repeat"]).mean().reset_index()

    # df = spatial_df.append(temporal_df)
    labels = spatial_df[target].to_numpy()
    spatial_data = spatial_df["correct"].to_numpy()
    temporal_data = temporal_df["correct"].to_numpy()
    df = pd.DataFrame({"spatial": spatial_data,
                       "temporal": temporal_data}, index=labels)

    df.plot.bar()
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel(target)

    plt.tight_layout()

    # save plt
    #plt.show()
    plt.savefig(os.path.join("analysis/fig/", args.output_file))
"""


