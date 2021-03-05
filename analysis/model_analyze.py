import pandas as pd
import numpy as np
import sys
import os

from sklearn.metrics import accuracy_score


def get_accuracy_c(df):

    #df["filename"] = df["filename"].str.split('/')[-1]
    #print(df)

    for mode in ["train", "evaluation"]:
        print('----'+mode.upper()+'----')
        df_mode = df[df["mode"] == mode]

        expected = df_mode["expected_label"]
        predicted = df_mode["predicted_label"]

        confusion_matrix = pd.crosstab(expected, predicted, rownames=['Expected'], colnames=['Predicted'])
        print(confusion_matrix)

        accuracy = accuracy_score(y_true=expected, y_pred=predicted)
        print("accuracy:", accuracy)


def get_accuracy_pl(df):
    timesteps = 12

    for mode in ["train", "evaluation"]:
        print('----'+mode.upper()+'----')
        df_mode = df[df["mode"] == mode]

        expected = np.concatenate([df_mode["expected_label_" + str(i)] for i in range(timesteps)])
        predicted = np.concatenate([df_mode["predicted_label_" + str(i)] for i in range(timesteps)])

        confusion_matrix = pd.crosstab(expected, predicted, rownames=['Expected'], colnames=['Predicted'])
        print(confusion_matrix)

        accuracy = accuracy_score(y_true=expected, y_pred=predicted)
        print("accuracy:", accuracy)


def get_accuracy_pl_obs(df):
    timesteps = 12

    for mode in ["train", "evaluation"]:
        print('----'+mode.upper()+'----')
        df_mode = df[df["mode"] == mode]

        num_traces = len(df_mode["obs_filename_0"])
        print("num_traces:", num_traces, "timesteps:", timesteps)
        obs = {}

        for i in range(num_traces):
            for j in range(timesteps):

                file_name = df_mode["obs_filename_"+str(j)][i].split('/')[-1]
                obs_name = file_name.split('_')[0]
                #print(file_name, obs_name)
                #print(i, j)

                if obs_name == 'r':
                    expected_0 = df_mode["expected_label_"+str(j)][i]
                    predicted_0 = df_mode["predicted_label_" + str(j)][i]
                    obs[file_name] = (expected_0 == predicted_0)

                elif obs_name == 'bg' or obs_name == 'gb' or obs_name == 'rr':
                    expected_0 = df_mode["expected_label_"+str(j)][i]
                    predicted_0 = df_mode["predicted_label_" + str(j)][i]
                    expected_1 = df_mode["expected_label_"+str(j)][i + 1]
                    predicted_1 = df_mode["predicted_label_" + str(j)][i + 1]
                    obs[file_name] = (expected_0 == predicted_0) and (expected_1 == predicted_1)

                elif obs_name == 'rrr':
                    expected_0 = df_mode["expected_label_" + str(j)][i]
                    predicted_0 = df_mode["predicted_label_" + str(j)][i]
                    expected_1 = df_mode["expected_label_" + str(j)][i + 1]
                    predicted_1 = df_mode["predicted_label_" + str(j)][i + 1]
                    expected_2 = df_mode["expected_label_" + str(j)][i + 2]
                    predicted_2 = df_mode["predicted_label_" + str(j)][i + 2]
                    obs[file_name] = (expected_0 == predicted_0) and (expected_1 == predicted_1) and (expected_2 == predicted_2)
        print("sort1")

        a_dict = {}
        for k, v in obs.items():
            obs_id = k.split('_')[0]
            if obs_id not in a_dict:
                a_dict[obs_id] = (0,0)
            a_dict[obs_id][0] += v
            a_dict[obs_id][1] += 1
        print("sort2")

        print("obs_accuracy:")
        for k, v in a_dict.items():
            print(k, v[0] / v[1])


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    parser.add_argument('app', help='classifier(c)/policy_learner(pl)', choices=['c', 'pl', 'pl_obs'])
    parser.add_argument('model', help='model_location')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_exec_args()
    df = pd.read_csv(os.path.join(args.model, "results.csv"))

    if args.app == "c":
        get_accuracy_c(df)
    elif args.app == "pl":
        get_accuracy_pl(df)
    else:
        get_accuracy_pl_obs(df)
