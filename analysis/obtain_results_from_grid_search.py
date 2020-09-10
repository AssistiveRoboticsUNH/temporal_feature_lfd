import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

import matplotlib
matplotlib.use('GTK')


if __name__ == "__main__":
    # load spatial data
    spatial_df = pd.read_csv(sys.argv[1])
    spatial_df["model"] = ["spatial"] * len(spatial_df)

    # load temporal data
    temporal_df = pd.read_csv(sys.argv[2])
    temporal_df["model"] = ["temporal"] * len(temporal_df)

    df = spatial_df.append(temporal_df)
    df["correct"] = df["expected_action"] == df["observed_action"]

    # group data by label
    df = df.groupby(["model", "bottleneck"]).mean()
    print(df)

    # generate graph of accuracy
    df.plot.hist(["correct"])
    plt.show()

    # save plt
