from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


class IADDataset(Dataset):

    def __init__(self, examples):
        super().__init__()

        # self.data = []  # contains tuples (iad, label)
        """
        for n in range(3):
            for s in range(examples):
                iad = np.zeros((3, 16))
                iad[n, np.random.randint(16, size=1)] = 1
                self.data.append((iad, n))
        """
        self.data = [
            (np.array([1, 0, 0]), 0),
            (np.array([0, 1, 0]), 1),
            (np.array([0, 0, 1]), 2)
        ]
        """
        self.data = [
            (np.array([1, 0, 0]), 0),
            (np.array([1, 0, 0]), 1),
            (np.array([1, 0, 0]), 2),

            (np.array([0, 1, 0]), 3),
            (np.array([0, 1, 1]), 4),
            (np.array([0, 1, 1]), 5),
            (np.array([0, 0, 1]), 6)
        ]
        """
        self.num_classes = len(self.data)

        block_size = 10
        new_data = []
        for data, label in self.data:
            img = np.zeros((block_size, block_size, 3), np.uint8)
            for j in range(3):
                if data[j]:
                    img[..., j] = 255
                    #img[j, :, block_size*j:block_size*j+block_size] = 255
            img = Image.fromarray(img)
            new_data.append((img, label))
        self.data = new_data

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        data, label = self.data[index][0], self.data[index][1]
        #print("data1.shape:", data.height, data.width)

        data = self.transform(data)
        #print("data2.shape:", data.shape)

        return data, label

    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.debug = False

        input_dims = 3  # * 16
        self.conv = nn.Sequential(
            #nn.Conv2d(3, 3, 1),  # in 3, out 3, size 3x3
            nn.Conv2d(3, 3, 3),
            #nn.Linear(3, 3),
            nn.ReLU(),
        )

        #self.pool = nn.AvgPool2d()

        self.linear = nn.Sequential(
            nn.Linear(input_dims, num_classes),
            nn.Tanh()
        )

    def forward(self, x):
        #print("x0.shape:", x.shape)
        # x = x.view((-1, 3, 16))
        # x, _ = torch.max(x, 2)
        # print(x)
        x = self.conv(x)
        #print("x1.shape:", x.shape)
        x = x.view(x.shape[0], 3, -1)
        #print("x2.shape:", x.shape)
        # print("x.shape:", x.shape)
        x, _ = torch.max(x, axis=2)
        if self.debug:
            print(x)
        # x = self.pool(x)  # try and see if there is an issue with one or the other


        return self.linear(x)


if __name__ == "__main__":
    train_dataset = IADDataset(5)
    test_dataset = IADDataset(5)

    batch_size = 1 # 3

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50)

    net = Model(num_classes=train_dataset.num_classes)
    net.debug = True

    params = list(net.parameters())
    net = nn.DataParallel(net, device_ids=[0]).cuda()
    net.train()

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

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
                print("label:", label)

                data = data.float().cuda()
                label = label.cuda()

                #print("data:", data.shape)

                logits = net(data)

                loss = criterion(logits, label)
                cummulative_loss += loss
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()

            print("loss:", cummulative_loss)
            cummulative_loss_arr.append(cummulative_loss)


    correct = 0
    for i, data_packet in enumerate(test_loader):
        data, label = data_packet[0], data_packet[1]
        data = data.float().cuda()

        logits = net(data)
        predicted = logits.detach().cpu().numpy()
        predicted = np.argmax(predicted, axis=1)

        print("pred:", predicted, "exp:", label)
        print("logits:")
        print(logits)

    plt.plot(cummulative_loss_arr)
    plt.show()
