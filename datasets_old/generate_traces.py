import numpy as np
import random

#n= 0, r =1, rr=2, rrr=3, g = 4, gb=5, bg = 6, b=7
#N=0, R=1, B=2, G=3

num_obs = 8
num_act = 4
NUM_TRACES = 100

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
    act = [1,1,1,2,2,2,3,3,3] + [0]*(length-9)
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
    act = [1,1,1,2,2,2,3,3,3] + [0]*(length-9)
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
def gen_path4(length=5):
    act = [0,0,0,1,1,1,2,2,2,3,3,3]
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

    print("obs:", obs)
    print("act:", act)



    return obs, act




# generate traces
dataset = []
for i in range(NUM_TRACES):
    obs, act = gen_path4()

    print("obs:", obs)
    print("act:", act)
    print("")

    eg = np.stack([obs, act])
    dataset.append(eg)
dataset = np.stack(dataset)

#print("dataset.shape:", dataset.shape)

# save files
np.save("/home/mbc2004/datasets/BlockConstruction/traces4.npy", dataset)

