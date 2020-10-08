from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import os
import math
import random

data_src = [
    ([1, 0, 2, 0], [1, 0, 1, 1]),
    ([1, 0, 0, 2, 0], [1, 0, 0, 1, 1]),
    ([1, 2, 0], [1, 1, 1]),
    ([1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),

    ([2, 0, 1], [1, 1, 1]),
    ([2, 0, 0, 1], [1, 1, 0, 1]),

    ([3, 0, 0], [1, 1, 1]),
    ([0, 3, 0, 0], [0, 1, 1, 1]),
    ([0, 0, 3, 0, 0], [0, 0, 1, 1, 1]),
]
random.shuffle(data_src)


class PolicyDataset(Dataset):

    def __init__(self, mode):
        super().__init__()

        self.data = data_src
        '''
        if mode == "train":
            self.data = data_src[:len(data_src) / 3]
        else:
            self.data = data_src[len(data_src) / 3:]
        '''

    def __getitem__(self, index):
        data, label = self.data[index][0], self.data[index][1]

        correct_data = np.zeros(3, len(data))
        for i in range(len(data)):
            correct_data[data[i]] = 1

        correct_label = np.zeros(2, len(label))
        for i in range(len(label)):
            correct_data[label[i]] = 1

        return correct_data, correct_label

    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.debug = False

        self.hidden_dim = 1
        self.lstm = nn.LSTM(3, 10, 2)

    def forward(self, x):
        x = self.lstm(x)
        return x


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
    criterion = nn.CrossEntropyLoss()#.cuda()

    # define optimizer
    optimizer = torch.optim.SGD(params,
                                0.001,
                                momentum=0.9,
                                weight_decay=0.005)
    epochs = 2000
    cummulative_loss_arr = []
    with torch.autograd.detect_anomaly():
        for e in range(epochs):
            cummulative_loss = 0
            for i, data_packet in enumerate(train_loader):
                data, label = data_packet[0], data_packet[1]

                data = data
                label = label

                logits = net(data)

                loss = criterion(logits, label)
                cummulative_loss += loss
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()

            print("loss:", cummulative_loss)
            cummulative_loss_arr.append(cummulative_loss)

    '''
    correct = 0
    for i, data_packet in enumerate(test_loader):
        data, label = data_packet[0], data_packet[1]
        data = data.float()#.cuda()

        logits = net(data, record =True)
        predicted = logits.detach().cpu().numpy()
        predicted = np.argmax(predicted, axis=1)

        print("pred:", predicted, "exp:", label)
        print("logits:")
        print(logits)

    plt.plot(cummulative_loss_arr)
    plt.show()

    min_v = 0
    max_v = 0

    for i, data_packet in enumerate(test_loader):
        data, label = data_packet[0], data_packet[1]
        data = data

        am = net(data, view=True)
        values = am.detach().cpu().numpy()
        print(values)
        if values.min() < min_v:
            min_v = values.min()
        if values.max() > max_v:
            max_v = values.max()

    print("min:", min_v, "max:", max_v)
    '''

    '''
    for i, data_packet in enumerate(test_loader):

        if i == 0 or i == len(test_loader)-1:
            data, label = data_packet[0], data_packet[1]
            data = data.float()  # .cuda()

            am = net(data, view=True)

            print("am:", am.shape)
            dim = int(math.sqrt(am.shape[-1]))
            am = am.view([1, 1, dim, dim]).detach().cpu().numpy()[0,0] * 254
            am = am.astype(np.uint8)
            img = Image.fromarray(am)


            img2 = (data.detach().cpu().numpy()[0] + 1) * 255
            print(img2.shape)
            img2 = img2.transpose([1,2,0])
            img2 = img2.astype(np.uint8)
            img2 = Image.fromarray(img2)

            lg_img = Image.new('RGB',(448, 224))
            lg_img.paste(img, (0,0))
            lg_img.paste(img2, (224, 0))

            lg_img.show()


            print("am.shape:", am.shape, am.min(), am.max())
            print("params:")
            print(params)

            input("wait")
    '''
