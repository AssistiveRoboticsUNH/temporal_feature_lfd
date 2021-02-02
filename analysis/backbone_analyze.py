import pandas as pd
import numpy as np
import os

def locate_files(src_dir, model):

    files = []
    for r, d, f in os.walk(src_dir):
        if f == "results.csv":
            files.append((d, file_path))
    return files

def organize_data(files):
    for f in files:
        print(f)

if __name__ == '__main__':

    src_dir = "."
    model = "i3d"

    files = locate_files(src_dir, model)
    organize_data(files)