import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def breakdown_full(spatial_df, title="", output_filename=""):
    spatial_df["model"] = ["spatial"] * len(spatial_df)
    spatial_df["correct"] = spatial_df["expected_label"] == spatial_df["predicted_label"]
    spatial_df["correct"] = spatial_df["correct"].astype(int)

    print("spatial_df:")
    print(spatial_df)

    spatial_df = spatial_df.groupby(["mode", "repeat", "expected_label"]).mean().reset_index()

    #spatial_df_mean = spatial_df.groupby(["mode", target, "expected_action"]).mean().reset_index()
    # spatial_df_std = spatial_df.groupby([target, "expected_action"]).std()#.reset_index()

    #label_order = [1, 2, 0]  # RGB
    label_order = [0, 1, 2, 3, 4, 5, 6, 7]  #  all classes
    label_dict_key = {"n": 0, "r": 1, "rr": 2, "rrr": 3, "g": 4, "gb": 5, "bg": 6, "b": 7}
    label_dict = {}  # matches labels in block construction
    for k in label_dict_key.keys():
        label_dict_key[label_dict_key[k]] = k
    colors = {"n": "black","r": "r", "g": "g", "b": "b", "bg": "cyan", "gb": "springgreen", "rr": "indianred", "rrr": "brown"}

    columns = {}
    for k in label_order:
        columns[label_dict[k]] = spatial_df.loc[spatial_df['predicted_label'] == k]["correct"].to_numpy()

    # label data
    modes = spatial_df["mode"].unique()
    repeats = spatial_df["repeat"].unique()
    labels = [m+", "+str(r) for m in modes for r in repeats]
    df = pd.DataFrame(columns, index=labels)

    print("columns:")
    print(columns)

    df.plot.bar(color=colors)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")

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
    parser.add_argument('--output_file', default="", help='the checkpoint file to use with the model')
    args = parser.parse_args()

    output_filename_acc = "" if args.output_file == "" else os.path.join("analysis/fig/", args.output_file + "_accuracy.png")
    output_filename_bd = "" if args.output_file == "" else os.path.join("analysis/fig/", args.output_file + "_breakdown.png")

    # load spatial data
    spatial_df_src = pd.read_csv(args.spatial_results_file)

    # run analysis
    breakdown_full(spatial_df_src, title=target_label+" Breakdown", output_filename=output_filename_bd)



