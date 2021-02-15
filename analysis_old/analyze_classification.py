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


def get_accuracy_policy_learning(df, timesteps):
    expected = np.concatenate([df["expected_label_"+str(i)] for i in range(timesteps)])
    predicted = np.concatenate([df["predicted_label_"+str(i)] for i in range(timesteps)])

    print(expected)
    print(predicted)

    return accuracy_score(y_true=expected, y_pred=predicted )


def view_accuracy(df, filename):
    print(df)
    print(df["obs_filename_0"])
    df["filename"] = df["obs_filename_0"].str.split('/').str[-1]
    print(df["filename"])

    df["obs_label"] = df["filename"].str.split('_').str[0]
    print(df["obs_label"])

    obs_list = ['None', 'r', 'rr', 'rrr', 'g', 'gb', 'bg', 'b']

    obs = []
    time = []
    action = []
    for i in range(len(df)):
        row = df.iloc[i]
        obs.extend([row["obs_label"]]*3)
        time.extend([0, 1, 2])

        action.append(row["predicted_label_0"])
        action.append(row["predicted_label_1"])
        action.append(row["predicted_label_2"])


    print("len(obs), len(time), len(action):", len(obs), len(time), len(action))

    new_df = pd.DataFrame({"obs": obs, "time": time, "action": action}, dtype="category")

    count_matrix = np.zeros((len(obs_list), 3, 4))
    for i in range(len(new_df)):
        row = new_df.iloc[i]

        o = obs_list.index(row["obs"])
        t = row["time"]
        a = row["action"]

        count_matrix[o, t, a] += 1

    obs = []
    time = []
    action = []
    percent = []

    for o in range(len(obs_list)):
        for t in [0,1,2]:
            for a in [0,1,2,3]:
                obs.append(obs_list[o])
                time.append(t)
                action.append(a)
                percent.append(float(count_matrix[o, t, a]) / np.sum(count_matrix[o, t]) )

    percent_df = pd.DataFrame({"obs": obs, "time": time, "action": action, "percent": percent})
    print(percent_df)


    fig, axs = plt.subplots(8, 3)
    for o in range(len(obs_list)):
        for t in [0, 1, 2]:
            data = percent_df[percent_df["obs"] == obs_list[o]]
            data = data[data["time"] == t]
            data = data["percent"]
            print(data)
            #pd.DataFrame(data).plot.bar()
            axs[o, t].bar(["N", "R", "G", "B"], data, color=["black", "red", "green", "blue"])

            axs[o, t].set_ylim(0, 1.0)
            #if t != 0:
            axs[o, t].get_yaxis().set_ticks([])
            if o != len(obs_list)-1:
                axs[o, t].get_xaxis().set_ticks([])

    if filename:
        plt.savefig(filename)
    plt.show()


def analyze_policy(df_filename, save_filename):
    df = pd.read_csv(df_filename)
    view_accuracy(df, save_filename)
    print("acc:", get_accuracy_policy_learning(df, timesteps=3))

if __name__ == '__main__':

    import sys
    input_files = [sys.argv[1]]
    for f in input_files:
        df = pd.read_csv(f)
        view_accuracy(df, f)

    print("acc:", get_accuracy_policy_learning(df, timesteps=3))

