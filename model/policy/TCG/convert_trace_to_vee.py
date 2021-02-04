import os
import numpy as np

src_dir = "/home/mbc2004/datasets/BlockConstruction"
trace_file = "traces6.npy"

if __name__ == '__main__':
    trace_path = os.path.join(src_dir, trace_file)
    traces = np.load(trace_path)

    print("traces.shape:", traces.shape)
    for i in range(traces.shape[0]):
        trace = traces[i]
        print("obs:", trace[0])
        print("act:", trace[1])
        print('')