from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import math
import random




num_obs = 8
num_act = 4

#n= 0, r =1, rr=2, rrr=3, g = 4, gb=5, bg = 6, b=7

#N=0, R=1, B=2, G=3

def gen_path(length=6):
    act = np.zeros(length, dtype=np.int)

    r_idx = np.random.choice(np.arange(length), 5, replace=False)
    r_idx = np.sort(r_idx)

    b_idx = np.random.choice(r_idx, 1)
    r_idx = r_idx[np.where(r_idx != b_idx)]
    g_idx = np.random.choice(r_idx, 1)
    r_idx = r_idx[np.where(r_idx != g_idx)]

    #print("r_idx:", r_idx)
    #print("b_idx:", b_idx)
    #print("g_idx:", g_idx)

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

    #print("obs:", obs)
    #print("act:", act)
    #assert False

    return obs, act


dataset = []
data = []
out_data = []
'''
for preceeding_zero in range(5):
    for following_zeros in range(2, 4):
        obs = [0] * preceeding_zero + [1] + [0] * following_zeros
        act = [0] * preceeding_zero + [1] * 3 + [0] * (following_zeros - 2)

        dataset.append((obs, act))
'''
for i in range(100):
    obs, act = gen_path()
    dataset.append((obs, act))


print("dataset:", len(dataset))

for obs, act in dataset:
    for idx in range(1, len(obs)):
        print(idx)
        print(obs)
        print(act)
        print(obs[:idx])
        print(act[:idx])
        print('')
        data.append((obs[:idx], act[:idx]))


print("data:", len(data))

for obs, act in data:
    data = np.zeros((num_obs + num_act, len(obs)))
    for i in range(len(obs)):
        print(data.shape, obs[i], i)

        data[obs[i], i] = 1
        data[num_obs + act[i], i] = 1

    data[num_obs:, len(obs) - 1] = 0

    label = act[-1]

    # print("data:", data.shape)
    # print("label:", label.shape)

    out_data.append((data, label))
random.shuffle(out_data)


class PolicyDataset(Dataset):

    def __init__(self, mode):
        super().__init__()

        if mode == "train":
            self.out_data = out_data[:- int(len(out_data)/3)]
        else:
            self.out_data = out_data[- int(len(out_data) / 3):]

    def __getitem__(self, index):

        data, label = self.out_data[index][0], self.out_data[index][1]
        #print("data:", data)
        #print("label:", label)

        data = data.T

        return torch.tensor(data), torch.tensor(label)


        """
        print("index:", index)
        index = int(index/2)
        data, label = self.data[index][0], self.data[index][1]

        correct_data = np.zeros((4, len(data)))
        for i in range(len(data)):
            # print("i:", i, "data[i]:", data[i], "len:", len(data), data)
            correct_data[data[i], i] = 1
        correct_data = correct_data.T

        correct_label = np.array(label)
        print("correct_data:", correct_data.shape)
        print("correct_label1:", correct_label.shape)

        idx = len(data)-1#index%2 #random.randint(1, len(data))

        #print("idx:", idx)s

        correct_data = correct_data[:idx+1]
        correct_label = correct_label[-1]


        print("correct_data:", correct_data.shape)
        print("correct_label2:", correct_label.shape)

        #correct_label = correct_label.reshape([1, 2])

        #print("correct_label3:", correct_label.shape)

        return torch.tensor(correct_data), torch.tensor(correct_label)
        """

    def __len__(self):
        return len(self.out_data)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.debug = False

        self.hidden_dim = 1
        #self.lstm = nn.LSTM(4, 2, 2)


        self.input_size=num_obs + num_act  # 4
        self.hidden_size=5
        self.num_layers=1#2
        self.num_classes=num_act

        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        #self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
        #                    num_layers=self.num_layers, batch_first=True)

        #self.lstm1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        #self.lstm2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.sft = torch.nn.Softmax()

    def forward(self, x):

        # print("x.shape:", x.shape)

        #out, hidden = self.lstm(x)

        # print("out:", out)
        # print("hidden:", hidden)


        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))


        #h_0 = Variable(torch.zeros(x.size(0), self.hidden_size))
        #c_0 = Variable(torch.zeros(x.size(0), self.hidden_size))
        #h_1 = Variable(torch.zeros(x.size(0), self.hidden_size))
        #c_1 = Variable(torch.zeros(x.size(0), self.hidden_size))

        '''
        output = []
        for input_t in x.split(1, dim=1):
            print("input_t1:", input_t.shape)
            #input_t = input_t.view([4])
            print("input_t2:", input_t.shape)
            out, (h_out, _) = self.lstm1(input_t, (h_0, c_0))
            #out, (h_out, _) = self.lstm2(out, (h_1, c_1))
            output += self.fc(out)
        #h_out = h_out.view(-1, self.hidden_size)
        #out = self.fc(h_out)
        #print("output.shape:", output[0].shape)
        outputs = torch.stack(output)
        outputs = outputs.view([1, -1, 2])
        '''

        out, (h_out, _) = self.lstm1(x, (h_0.detach(), c_0.detach()))
        print("out1.shape:", out.shape)
        #print("out:", out)
        out = self.fc(out)
        print("out2.shape:", out.shape)
        #print("out2:", out)
        #out = self.sft(out)
        out = out[:, -1, :]
        #out = F.log_softmax(out, dim=1)
        print("out3.shape:", out.shape)

        return out


if __name__ == "__main__":

    train_dataset = PolicyDataset("train")
    test_dataset = PolicyDataset("evaluation")

    batch_size = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    net = Model()
    net.debug = True

    params = list(net.parameters())
    net.train()

    # define loss function
    criterion = nn.CrossEntropyLoss()#.cuda() nn.NLLLoss() #

    # define optimizer
    optimizer = torch.optim.SGD(params,
                                0.001,
                                momentum=0.9,
                                weight_decay=0.005)
    epochs = 200
    cummulative_loss_arr = []
    with torch.autograd.detect_anomaly():
        for e in range(epochs):
            cummulative_loss = 0
            for i, data_packet in enumerate(train_loader):
                data, label = data_packet[0], data_packet[1]

                data = data.float()
                label = label#.float()

                print("data:", data.shape)
                print("label:", label.shape)

                logits = net(data)

                #print("label:", label.shape)
                #print("label:", label)
                print("logits:", logits.shape)

                loss = criterion(logits, label)
                cummulative_loss += loss
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()

            print("e:", e, "loss:", cummulative_loss)
            cummulative_loss_arr.append(cummulative_loss)


    correct = 0
    for i, data_packet in enumerate(train_loader):
        data, label = data_packet[0], data_packet[1]

        data = data.float()
        label = label#.float()

        logits = net(data)
        predicted = logits.detach().cpu().numpy()

        pred_max = np.argmax(predicted, axis=1)[0]
        #print("pred_max:")
        #print(pred_max)

       # pred_max2 = np.zeros((2, len(pred_max)))
        #for i in range(len(pred_max)):
        #    pred_max2[pred_max[i], i] = 1
        #pred_max2 = pred_max2.T

        #print("pred_max2")
        #print(predicted)



        print("pred:")
        print(pred_max)
        print("exp:")
        print(label.detach().cpu().numpy())
        print('')
        print("logits:")
        print(logits)

    plt.plot(cummulative_loss_arr)
    plt.show()

