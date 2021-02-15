import pandas as pd
import numpy as np
import sys
import os

from sklearn.metrics import confusion_matrix, accuracy_score


def get_accuracy_c(df):

    df["filename"] = df["filename"].str.split('/')[-1]
    print(df)

    for mode in ["train", "evaluation"]:
        df_mode = df[df["mode"] == mode]

        expected = df_mode["expected_label"]
        predicted = df_mode["predicted_label"]

        accuracy = accuracy_score(y_true=expected, y_pred=predicted)
        print(mode, accuracy)


def get_accuracy_pl(df):
    timesteps = 3

    print(df)

    expected = np.concatenate([df["expected_label_" + str(i)] for i in range(timesteps)])
    predicted = np.concatenate([df["predicted_label_" + str(i)] for i in range(timesteps)])
    accuracy = accuracy_score(y_true=expected, y_pred=predicted)
    print(accuracy)


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    parser.add_argument('app', help='classifier(c)/policy_learner(pl)', choices=['c', 'pl'])
    parser.add_argument('model', help='model_location')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_exec_args()
    df = pd.read_csv(os.path.join(args.model, "results.csv"))

    if args.app == "c":
        get_accuracy_c(df)
    else:
        get_accuracy_pl(df)
