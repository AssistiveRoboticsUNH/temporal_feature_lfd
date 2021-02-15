import pandas as pd
import numpy as np
import sys
import os

from sklearn.metrics import confusion_matrix, accuracy_score


def get_accuracy(df):
    timesteps = 3

    print(df)

    expected = np.concatenate([df["expected_label_" + str(i)] for i in range(timesteps)])
    predicted = np.concatenate([df["predicted_label_" + str(i)] for i in range(timesteps)])
    # print(pd.DataFrame({"expected": expected, "predicted:": predicted}))
    return accuracy_score(y_true=expected, y_pred=predicted)


if __name__ == '__main__':

    model = sys.argv[1]
    df = pd.read_csv(os.path.join(model, "results.csv"))

    get_accuracy(df)
