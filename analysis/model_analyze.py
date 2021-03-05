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
    timesteps = 3

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
    timesteps = 3

    for mode in ["train", "evaluation"]:
        print('----'+mode.upper()+'----')
        df_mode = df[df["mode"] == mode]

        print(df_mode)

        obs_filenames = np.concatenate([df_mode["obs_filename_" + str(i)] for i in range(timesteps)])
        obs_filenames = obs_filenames.str.split('/')
        print(obs_filenames)
        expected = np.concatenate([df_mode["expected_label_" + str(i)] for i in range(timesteps)])
        predicted = np.concatenate([df_mode["predicted_label_" + str(i)] for i in range(timesteps)])

        confusion_matrix = pd.crosstab(expected, predicted, rownames=['Expected'], colnames=['Predicted'])
        print(confusion_matrix)

        accuracy = accuracy_score(y_true=expected, y_pred=predicted)
        print("accuracy:", accuracy)




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
