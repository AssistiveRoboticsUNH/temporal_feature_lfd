import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    # load spatial data
    spatial_df = pd.read_csv(sys.argv[1])
    spatial_df["model"] = ["spatial"]*len(spatial_df)

    # load temporal data
    # temporal_df = pd.read_csv(sys.argv[2])
    # temporal_df["model"] = ["temporal"] * len(temporal_df)

    #df = spatial_df.append(temporal_df)
    df = spatial_df
    df["correct"] = df["expected_action"] == df["observed_action"]

    # group data by label
    df = df.group_by(["model", "bottleneck"])

    # generate graph of accuracy

    # save plt
