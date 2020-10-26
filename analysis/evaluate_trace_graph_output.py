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

    new_df = pd.DataFrame({"obs_label": df["obs_label"],
                           "expected_label_0": df["expected_label_0"],
                           "expected_label_1": df["expected_label_1"],
                           "expected_label_2": df["expected_label_2"]})
    print(new_df)



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
