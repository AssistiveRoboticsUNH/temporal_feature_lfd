import os
import numpy as np

src_dir = "/home/mbc2004/datasets/BlockConstruction"
trace_file = "traces6.npy"

if __name__ == '__main__':
    trace_path = os.path.join(src_dir, trace_file)
    traces = np.load(trace_path)

    obs_arr = ['r', 'rr', 'rrr', 'b', 'bg', 'gb', 'g', 'n']
    act_arr = ['R', 'G', 'B', 'N']

    print("traces.shape:", traces.shape)
    for i in range(traces.shape[0]):
        trace = traces[i]
        obs = trace[0]
        act = trace[1]

        file_dir = os.path.join(src_dir, 'vee_trace')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        filename = os.path.join(file_dir, str(i).zfill(3)+".txt")
        ofile = open(filename, 'w')

        counter = 0

        for t in range(traces.shape[-1]):
            ofile.write(obs_arr[obs[t]] + '_0' + '_s %.1f\n' % counter)
            counter += 1
            ofile.write(obs_arr[obs[t]] + '_0' + '_e %.1f\n' % counter)
            counter += 1

            ofile.write(act_arr[act[t]] + '_0' + '_s %.1f\n' % counter)
            counter += 1
            ofile.write(act_arr[act[t]] + '_0' + '_e %.1f\n' % counter)
            counter += 1

        ofile.close()
