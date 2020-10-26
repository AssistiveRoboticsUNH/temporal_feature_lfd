import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import argparse
import os

from sklearn.metrics import confusion_matrix, accuracy_score


def get_accuracy(df):
    expected = df["expected_label"]
    observed = df["predicted_label"]

    return accuracy_score(y_true = expected, y_pred = observed )


def view_accuracy(df):
    print(df)
    print(df["obs_filename_0"])
    df["filename"] = df["obs_filename_0"].str.split('/').str[-1]
    print(df["filename"])

    df["obs_label"] = df["filename"].str.split('_').str[0]
    print(df["obs_label"])

    obs = []
    time = []
    action = []
    for i in range(len(df)):
        row = df.iloc[i]
        print("row:", row, row["obs_label"])
        obs.append(row["obs_label"])
        time.extend([0, 1, 2])

        action.append(row["predicted_label_0"])
        action.append(row["predicted_label_1"])
        action.append(row["predicted_label_2"])

    print("len(obs), len(time), len(action):", len(obs), len(time), len(action))

    new_df = pd.DataFrame({"obs": obs, "time": time, "action": action})

    """
    '''
    new_df = pd.DataFrame({"obs": df["obs_label"],
                           "predicted_label_0": df["predicted_label_0"],
                           "predicted_label_1": df["predicted_label_1"],
                           "predicted_label_2": df["predicted_label_2"]})
    '''
    new_df = pd.DataFrame({"obs": df["obs_label"],
                           "expected_label_0": df["expected_label_0"],
                           "expected_label_1": df["expected_label_1"],
                           "expected_label_2": df["expected_label_2"]}, dtype="category")
    """
    #new_df = new_df[new_df["obs"] == 'rrr']["expected_label_0"]
    print(new_df)
    #print(new_df.groupby("obs").mean())
    new_df.plot.bar()
    plt.show()



if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Generate IADs from input files')
    parser.add_argument('input_file', help='the checkpoint file to use with the model')
    parser.add_argument('--fig_dir', default="analysis/fig", help='the checkpoint file to use with the model')
    args = parser.parse_args()

    src_filename = args.input_file.split("/")[-1][:-4]

    fig_dir = os.path.join(args.fig_dir, src_filename)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    '''
    input_file = "csv_output/output_policy_learning_ditrl_i3d_action_trace_ablation.csv"

    #df = pd.read_csv(args.input_file)
    df = pd.read_csv(input_file)

    #acc = get_accuracy(df)
    view_accuracy(df)
    # viz_confusion_matrix(df, os.path.join(fig_dir, "cm.png"))

    #print("filename: ", src_filename)
    #print("accuracy: ", acc)
