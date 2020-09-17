import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def breakdown(input_df, target, title="", output_filename=""):
    input_df["filename_short"] = input_df["filename"].str.split('/').str[-1]
    print(input_df["filename_short"])
    input_df["correct"] = input_df["expected_action"] == input_df["observed_action"]
    spatial_df = input_df.groupby([target, "repeat", "expected_action"]).mean().reset_index()

    spatial_df_mean = spatial_df.groupby([target, "expected_action"]).mean().reset_index()
    # spatial_df_std = spatial_df.groupby([target, "expected_action"]).std()#.reset_index()

    label_order = [1, 3, 4, 2, 0, 5, 6]
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

    cases = input_df[input_df["bottleneck"] == 4]
    print("print(cases)1:", cases)
    cases = cases[~cases["correct"]]
    print("print(cases)2:", cases)
    cases = cases["filename_short"].reset_index()
    print("print(cases)3:", cases)
    print("")
    print(cases.groupby(["filename_short"]).count())


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
    #parser.add_argument('temporal_results_file', help='the checkpoint file to use with the model')
    parser.add_argument('--output_file', default="", help='the checkpoint file to use with the model')
    args = parser.parse_args()

    output_filename_acc = "" if args.output_file == "" else os.path.join("analysis/fig/", args.output_file + "_accuracy.png")
    output_filename_bd = "" if args.output_file == "" else os.path.join("analysis/fig/", args.output_file + "_breakdown.png")

    # load spatial data
    spatial_df_src = pd.read_csv(args.spatial_results_file)
    spatial_df_src["model"] = ["spatial"] * len(spatial_df_src)

    #temporal_df_src = pd.read_csv(args.spatial_results_file)

    # run analysis
    #accuracy(spatial_df_src, temporal_df_src, target_label, title=target_label+" Accuracy", output_filename=output_filename_acc)
    breakdown(spatial_df_src, target_label, title=target_label+" Breakdown", output_filename=output_filename_bd)


