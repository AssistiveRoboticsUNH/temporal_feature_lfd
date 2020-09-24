import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def accuracy(spatial_df, target, title="", output_filename=""):
    spatial_df["model"] = ["spatial"] * len(spatial_df)
    spatial_df["correct"] = spatial_df["expected_action"] == spatial_df["observed_action"]
    spatial_df = spatial_df.groupby([target, "repeat"]).mean().reset_index()

    spatial_df_mean = spatial_df.groupby(["mode", target]).mean().reset_index()
    spatial_df_std = spatial_df.groupby(["mode", target]).std().reset_index()

    labels = spatial_df[target].unique()
    spatial_data_mean = spatial_df_mean["correct"].to_numpy()
    spatial_data_std = spatial_df_std["correct"].to_numpy()

    print(labels)
    print(spatial_data_mean)
    print(spatial_data_std)

    df = pd.DataFrame({"spatial": spatial_data_mean}, index=labels)

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


def breakdown(spatial_df, target, title="", output_filename=""):
    spatial_df["model"] = ["spatial"] * len(spatial_df)
    spatial_df["correct"] = spatial_df["expected_action"] == spatial_df["observed_action"]
    spatial_df["correct"] = spatial_df["correct"].astype(int)

    print("spatial_df:")
    print(spatial_df)

    spatial_df = spatial_df.groupby(["mode", target, "repeat", "expected_action"]).mean().reset_index()

    #spatial_df_mean = spatial_df.groupby(["mode", target, "expected_action"]).mean().reset_index()
    # spatial_df_std = spatial_df.groupby([target, "expected_action"]).std()#.reset_index()
    #print("sd3:", spatial_df_mean)

    label_order = [1, 2, 0]  # RGB
    #label_order = [1, 3, 4, 2, 0, 5, 6]  #  all classes
    label_dict = {0: 'r', 1: 'g', 2: 'b', 3: 'gb', 4: 'bg', 5: 'rr', 6: 'rrr'}  # matches labels in block construction
    colors = {"r": "r", "g": "g", "b": "b", "bg": "cyan", "gb": "springgreen", "rr": "indianred", "rrr": "brown"}
    columns = {}
    for k in label_order:
        columns[label_dict[k]] = spatial_df.loc[spatial_df['expected_action'] == k]["correct"].to_numpy()

    action_labels = spatial_df[target].unique()
    modes = spatial_df["mode"].unique()
    repeats = spatial_df["repeat"].unique()
    labels = [m+", "+str(l)+", "+str(r) for m in modes for l in action_labels for r in repeats]
    df = pd.DataFrame(columns, index=labels)

    print("columns:")
    print(columns)

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
    parser.add_argument('--output_file', default="", help='the checkpoint file to use with the model')
    args = parser.parse_args()

    output_filename_acc = "" if args.output_file == "" else os.path.join("analysis/fig/", args.output_file + "_accuracy.png")
    output_filename_bd = "" if args.output_file == "" else os.path.join("analysis/fig/", args.output_file + "_breakdown.png")

    # load spatial data
    spatial_df_src = pd.read_csv(args.spatial_results_file)

    # run analysis
    #accuracy(spatial_df_src, target_label, title=target_label+" Accuracy", output_filename=output_filename_acc)
    breakdown(spatial_df_src, target_label, title=target_label+" Breakdown", output_filename=output_filename_bd)



