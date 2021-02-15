import pandas as pd
import numpy as np
import sys
import os

from sklearn.metrics import confusion_matrix, accuracy_score


def get_accuracy_c(df):

    print(df)

    expected = df["expected_label"]
    predicted = df["predicted_label"]

    return accuracy_score(y_true=expected, y_pred=predicted)

def get_accuracy_pl(df):
    timesteps = 3

    print(df)

    expected = np.concatenate([df["expected_label_" + str(i)] for i in range(timesteps)])
    predicted = np.concatenate([df["predicted_label_" + str(i)] for i in range(timesteps)])
    # print(pd.DataFrame({"expected": expected, "predicted:": predicted}))
    return accuracy_score(y_true=expected, y_pred=predicted)


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
