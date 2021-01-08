import numpy as np
import random

#n= 0, r =1, rr=2, rrr=3, g = 4, gb=5, bg = 6, b=7
#N=0, R=1, B=2, G=3

num_obs = 8
num_act = 4
NUM_TRACES = 200#1000

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


'''
generates paths but all of the observations line up with the actions. There is no backlog and every action can be 
determined from the last 3 turns
'''
def gen_path2(length=10):
    assert length <= 12
    act = [0,0,0,1,1,1,2,2,2,3,3,3]
    random.shuffle(act)
    act = np.array(act[:length])

    obs = np.zeros(length, dtype=np.int)

    j = 0
    while j < len(act):
        add = 1

        #print("s:", j)

        if act[j] == 1:
            #print("c1")
            obs[j] = 1
        if j < len(act)-1 and act[j] == 1 and act[j+1] == 1:
           # print("c2")
            obs[j] = 2
            obs[j+1] = 0
            add = 2
        if j < len(act)-2 and act[j] == 1 and act[j+1] == 1 and act[j+2] == 1:
          #  print("c3")
            obs[j] = 3
            obs[j + 1] = 0
            obs[j + 2] = 0
            add = 3

        if act[j] == 2:
            #print("c4")
            obs[j] = 4
        if j < len(act)-1 and act[j] == 2 and act[j+1] == 3:
            #print("c5")
            obs[j] = 5
            obs[j + 1] = 0
            add = 2

        if act[j] == 3:
            #print("c6")
            obs[j] = 7
        if j < len(act)-1 and act[j] == 3 and act[j+1] == 2:
            #print("c7")
            obs[j] = 6
            obs[j + 1] = 0
            add = 2

        j += add
        #print("e", j)

    return obs, act

'''
just RGB and N perfectly matched to the actions
'''
def gen_path3(length=10): # only RGBN
    act = [1,1,1,2,2,2,3,3,3, 0, 0, 0]
    random.shuffle(act)
    act = np.array(act)

    obs = np.zeros(length, dtype=np.int)

    j = 0
    while j < len(act):
        add = 1

        #print("s:", j)

        if act[j] == 1:
            #print("c1")
            obs[j] = 1

        if act[j] == 2:
            #print("c4")
            obs[j] = 4

        if act[j] == 3:
            #print("c6")
            obs[j] = 7

        j += add
        #print("e", j)

    return obs, act


'''
all observations are up-front
'''
def obs_generator(length):
    obs_dict = {'n': [0], 'r': [1], 'rr': [2, 0], 'rrr': [3, 0, 0], 'g': [4], 'gb': [5, 0], 'bg': [6, 0], 'b': [7]}
    act_dict = {'n': [0], 'r': [1], 'rr': [1, 1], 'rrr': [1, 1, 1], 'g': [2], 'gb': [2, 3], 'bg': [3, 2], 'b': [3]}

    act_k = list(act_dict.keys())
    random.shuffle(act_k)

    # make sure I have enough blocks to pick
    colors_in_model = np.array([0,0,0,0])
    colors_limit = np.array([100, 3, 3, 3])
    for i in range(len(act_k)):
        color_in_action = np.array([0,0,0,0])
        for a in act_dict[act_k[i]]:
            color_in_action[a] += 1

        color_temp = colors_in_model + color_in_action
        if np.any(color_temp > colors_limit):
            # cannot pick blocks
            act_k[i] = 'n'
        else:
            colors_in_model = color_temp

    # generate observations
    obs, act = [], []
    for o in act_k:
        obs.extend(obs_dict[o])
        act.extend(act_dict[o])

    return np.array(obs[:length]), np.array(act[:length])


def gen_path4(length=5):
    obs, act = obs_generator(length)

    force_stops = [x for x in range(length) if act[x] == 0]
    force_stops.append(length)

    new_obs = np.zeros_like(obs)
    new_obs_idx = 0
    for i in range(length):
        if i in force_stops:
            new_obs_idx = i+1
        if obs[i] != 0:
            new_obs[new_obs_idx] = obs[i]
            new_obs_idx += 1

    return new_obs, act




np.random.seed(0)
random.seed(0)

# generate traces
dataset = []
for i in range(NUM_TRACES):
    obs, act = gen_path2(length=5)

    print("obs:", obs)
    print("act:", act)
    print("")

    eg = np.stack([obs, act])
    dataset.append(eg)
dataset = np.stack(dataset)

#print("dataset.shape:", dataset.shape)

# save files
np.save("/home/mbc2004/datasets/BlockConstruction/traces6.npy", dataset)

