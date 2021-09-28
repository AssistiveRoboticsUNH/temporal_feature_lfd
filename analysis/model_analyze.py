import pandas as pd
import numpy as np
import sys
import os

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=4)

def get_accuracy_c(args, df):

    if args.save_cm:
        dataset_path = input("Path to class labels: ")
        class_labels = sorted(os.listdir(dataset_path))

    for mode in ["train", "evaluation"]:
        print('----'+mode.upper()+'----')
        df_mode = df[df["mode"] == mode]

        expected = df_mode["expected_label"]
        predicted = df_mode["predicted_label"]

        norm = 'false' if args.normalize else 'true'
        cm = confusion_matrix(expected, predicted, normalize=norm)
        print(cm)

        accuracy = accuracy_score(y_true=expected, y_pred=predicted)
        print("accuracy:", accuracy)

        if args.save_cm:

            df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
            plt.figure(figsize=(10, 7))
            sns.set(font_scale=1.5)
            plot = sns.heatmap(df_cm, annot=True, cmap="YlGnBu", vmax=1)
            plt.tight_layout()

            plot.set_yticklabels(plot.get_yticklabels(), rotation=0)
            plot.figure.savefig(os.path.join(args.model,"cm_"+mode+".png"))


def get_accuracy_pl(args, df):
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


def get_accuracy_pl_obs(args, df):
    timesteps = 12

    for mode in ["train", "evaluation"]:
        print('----'+mode.upper()+'----')
        df_mode = df[df["mode"] == mode].reset_index()

        num_traces = len(df_mode["obs_filename_0"])
        print("num_traces:", num_traces, "timesteps:", timesteps)
        obs = {}

        for i in range(num_traces):
            for j in range(timesteps):

                print("obs_filename_"+str(j))
                print(df_mode["obs_filename_"+str(j)])
                print(i)
                print(df_mode["obs_filename_"+str(j)][i])
                print(df_mode["obs_filename_"+str(j)][i].split('/')[-1])

                file_name = df_mode["obs_filename_"+str(j)][i].split('/')[-1]
                obs_name = file_name.split('_')[0]
                print(i, j, obs_name)

                if obs_name == 'r':
                    expected_0 = df_mode["expected_label_"+str(j)][i]
                    predicted_0 = df_mode["predicted_label_" + str(j)][i]
                    obs[file_name] = (expected_0 == predicted_0)

                elif (obs_name == 'bg' or obs_name == 'gb' or obs_name == 'rr') and j < timesteps-1:
                    expected_0 = df_mode["expected_label_"+str(j)][i]
                    predicted_0 = df_mode["predicted_label_" + str(j)][i]
                    expected_1 = df_mode["expected_label_"+str(j+1)][i]
                    predicted_1 = df_mode["predicted_label_" + str(j+1)][i]
                    obs[file_name] = (expected_0 == predicted_0) and (expected_1 == predicted_1)

                elif obs_name == 'rrr' and j < timesteps-2:
                    expected_0 = df_mode["expected_label_" + str(j)][i]
                    predicted_0 = df_mode["predicted_label_" + str(j)][i]
                    expected_1 = df_mode["expected_label_" + str(j+1)][i]
                    predicted_1 = df_mode["predicted_label_" + str(j+1)][i]
                    expected_2 = df_mode["expected_label_" + str(j+2)][i]
                    predicted_2 = df_mode["predicted_label_" + str(j+2)][i]
                    obs[file_name] = (expected_0 == predicted_0) and (expected_1 == predicted_1) and (expected_2 == predicted_2)

        a_dict = {}
        for k, v in obs.items():
            obs_id = k.split('_')[0]
            if obs_id not in a_dict:
                a_dict[obs_id] = [0, 0]

            a_dict[obs_id][0] += v
            a_dict[obs_id][1] += 1

        print("obs_accuracy:")
        for k, v in a_dict.items():
            print(k, v[0] / v[1])


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    parser.add_argument('app', help='classifier(c)/policy_learner(pl)', choices=['c', 'pl', 'pl_obs'])
    parser.add_argument('model', help='model_location')

    parser.set_defaults(save_cm=False)
    parser.add_argument('--save_cm', help='save confusion_matrix', dest='save_cm', action='store_true')
    parser.set_defaults(normalize=False)
    parser.add_argument('--no_norm', help='stop normalizing results', dest='normalize', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_exec_args()
    df = pd.read_csv(os.path.join(args.model, "results.csv"))

    if args.app == "c":
        get_accuracy_c(args, df)
    elif args.app == "pl":
        get_accuracy_pl(args, df)
    else:
        get_accuracy_pl_obs(args, df)
