import pandas as pd
import numpy as np
import os


def locate_files(src_dir, model):

    files = []
    for r, d, f in os.walk(src_dir):
        print(r, d, f)
        for filename in f:
            if f == "results.csv":
                file_path = os.path.join(r, filename)
                run_name = r.split('/')[-1]
                files.append((run_name, file_path))
    return files


def organize_data(files):
    for f in files:
        print(f)

if __name__ == '__main__':

    src_dir = "."
    model = "i3d"

    files = locate_files(src_dir, model)
    organize_data(files)