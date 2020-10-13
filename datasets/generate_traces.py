import numpy as np
import random

#n= 0, r =1, rr=2, rrr=3, g = 4, gb=5, bg = 6, b=7
#N=0, R=1, B=2, G=3

num_obs = 8
num_act = 4
NUM_TRACES = 100

def gen_path2(length=10):
    act = [1,1,1,2,2,2,3,3,3] + [0]*(length-9)
    random.shuffle(act)
    act = np.array(act)

    obs = np.zeros(length, dtype=np.int)

    last_idx = 0
    for j in range(1, len(act)):
        odd_case = False
        obs_val = 0
        if act[last_idx] == 1 and act[j] != act[last_idx]: # R
            if j - last_idx == 3:
                obs_val = 3
            elif j - last_idx == 2:
                obs_val = 2
            elif j - last_idx == 1:
                obs_val = 1
        elif act[last_idx] == 2:# G
            if act[j] == 3:
                obs_val = 5
                odd_case = True
            else:
                obs_val = 4
        elif act[last_idx] == 3:# B
            if act[j] == 2:
                obs_val = 6
                odd_case = True
            else:
                obs_val = 7

        print(last_idx, obs_val)

        obs[last_idx] = obs_val
        last_idx = j
        if odd_case:
            last_idx = j + 1
            j += 2

    return obs, act



def gen_path(length=10):
    act = np.zeros(length, dtype=np.int)

    # get indexes of R action
    r_idx = np.random.choice(np.arange(length), 5, replace=False)
    r_idx = np.sort(r_idx)

    # get indexes of B action
    b_idx = np.random.choice(r_idx, 1)
    r_idx = r_idx[np.where(r_idx != b_idx)]
    # get indexes of G action
    g_idx = np.random.choice(r_idx, 1)
    r_idx = r_idx[np.where(r_idx != g_idx)]

    act[r_idx] = 1
    act[b_idx] = 2
    act[g_idx] = 3

    obs = np.zeros(length, dtype=np.int)

    # caluclate r_obs
    if r_idx[2] - r_idx[0] == 2:
        obs[r_idx[0]] = 3  # rrr
    elif r_idx[1]-r_idx[0] == 1:
        obs[r_idx[0]] = 2  # rr
        obs[r_idx[2]] = 1
    elif r_idx[2]-r_idx[1] == 1:
        obs[r_idx[1]] = 2  # rr
        obs[r_idx[0]] = 1
    else:
        obs[r_idx[2]] = 1  # r
        obs[r_idx[1]] = 1
        obs[r_idx[0]] = 1

    # calculate bg_obs
    if b_idx - g_idx == 1:
        obs[g_idx] = 5  # add gb
    elif g_idx - b_idx == 1:
        obs[b_idx] = 6  # add bg
    else:
        obs[b_idx] = 7  # add b
        obs[g_idx] = 4  # add g

    return obs, act


# generate traces
dataset = []
for i in range(NUM_TRACES):
    obs, act = gen_path2()

    print("obs:", obs)
    print("act:", act)
    print("")

    eg = np.stack([obs, act])
    dataset.append(eg)
dataset = np.stack(dataset)

#print("dataset.shape:", dataset.shape)

# save files
np.save("/home/mbc2004/datasets/BlockConstruction/traces2.npy", dataset)

