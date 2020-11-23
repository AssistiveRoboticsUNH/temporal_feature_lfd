import os
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np

np.random.seed(0)


def gen_data_duration_inv(length, label):
    idx = np.random.choice(list(range(length)), 2, replace=False)

    toggle = 0
    iad = np.zeros((length, 3))

    for i in range(length):
        if i in idx:
            if toggle == 0:
                toggle = 1
            else:
                toggle = 0
        #iad.append(toggle)
        iad[i, label] = toggle

    return iad#np.array(iad).reshape(-1, 1)


def gen_data_duration_dependencies(length, label):
    idx = np.random.choice(list(range(length)), 4, replace=False)

    toggle = 0
    iad = np.zeros((length, 3))

    if label == 0:
        ord = [0, 1]
    elif label == 1:
        ord = [1, 0]
    else:
        ord = [1, 2]


    idx_l = 0
    for i in range(length):
        if i in idx:
            if toggle == 0:
                toggle = 1
                if len(ord) > 0:
                    idx_l = ord.pop(0)
            else:
                toggle = 0

        iad[i, idx_l] = toggle

    return iad


def gen_data_cylical(length, label):
    idx = np.random.choice(list(range(length)), label*2 +1, replace=False)

    toggle = 0
    iad = np.zeros((length, 3))

    for i in range(length):
        if i in idx:
            if toggle == 0:
                toggle = 1
            else:
                toggle = 0
        iad[i, :] = toggle

    return iad


def gen_data_cylical_measured(length, label):
    idx = np.linspace(0, 20, label*2 )
    print("idx:", idx)

    toggle = 0
    iad = np.zeros((length, 3))

    for i in range(length):
        if i in idx:
            if toggle == 0:
                toggle = 1
            else:
                toggle = 0
        iad[i, :] = toggle

    return iad


data = []
for i in range(10):
    for l in range(3):
        inp = (gen_data_cylical_measured(20, l), l)
        data.append(inp)

        if i < 3:
            print("inp:", inp[0], inp[1])


class CustomDataset(Dataset):
    def __init__(self, mode):
        self.dataset = []
        split = int(len(data)*(2/3))
        print("Split:", split)
        if mode == "train":
            self.dataset = data[:split]
        else:
            self.dataset = data[split:]

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


class LSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        # constants params
        self.input_size = input_size
        self.hidden_size = 64
        self.num_layers = 1
        self.output_size = output_size

        # define model vars
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    # Defining the forward pass
    def forward(self, x):
        #print("spatial x.shape1:", x.shape)

        # create empty vars for LSTM
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))#.cuda()
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))#.cuda()

        # obtain logits
        x, (h_out, _) = self.lstm(x, (h_0.detach(), c_0.detach()))
        x = self.fc(x)
        x = x[:, -1, :]

        #print("spatial x.shape2:", x.shape)
        return x


if __name__ == "__main__":

    train_dataset = CustomDataset("train")
    test_dataset = CustomDataset("evaluation")

    batch_size = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    net = LSTM(3, 3)

    params = list(net.parameters())
   # net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    net.train()

    # define loss function
    criterion = nn.CrossEntropyLoss()#.cuda()# nn.NLLLoss() #

    # define optimizer
    optimizer = torch.optim.SGD(params,
                                0.001,
                                momentum=0.9,
                                weight_decay=0.005)
    epochs = 200
    cummulative_loss_arr = []
    with torch.autograd.detect_anomaly():
        for e in range(epochs):
            #print("e:", e, epochs)
            cummulative_loss = 0
            for i, data_packet in enumerate(train_loader):
                data, label = data_packet[0], data_packet[1]

                data = data.float()
                label = label#.float()

                #print("data:", data.shape)
                #print("label:", label.shape)

                logits = net(data)

                #print("label:", label.shape)
                #print("label:", label)
                #print("logits:", logits.shape)

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
        label = label.numpy()[0]   #.float()

        logits = net(data)
        predicted = logits.detach().cpu().numpy()
        pred_max = np.argmax(predicted, axis=1)[0]

        print(pred_max, label)
        if pred_max == label:
            correct += 1

    print("TRAIN Total Accuracy:", correct / len(train_loader))

    correct = 0
    for i, data_packet in enumerate(test_loader):
        data, label = data_packet[0], data_packet[1]

        data = data.float()
        label = label.numpy()[0]

        logits = net(data)
        predicted = logits.detach().cpu().numpy()
        pred_max = np.argmax(predicted, axis=1)[0]

        print(pred_max, label)
        if pred_max == label:
            correct += 1

    print("EVAL Total Accuracy:", correct / len(test_loader))
