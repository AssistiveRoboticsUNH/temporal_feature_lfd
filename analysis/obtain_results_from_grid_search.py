import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    # load spatial data
    spatial_df = pd.read_csv(sys.argv[1])

    # load temporal data
    temporal_df = pd.read_csv(sys.argv[2])

    # add labels (bottleneck size, and temporal v. spatial information)

    # group data by label
    # group_df = df.group_by("")

    # generate graph of accuracy

    # save plt
