from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np


class IADDataset(Dataset):

    def __init__(self, examples):
        super().__init__()

        self.data = []  # contains tuples (iad, label)
        for n in range(3):
            for s in range(examples):
                iad = np.zeros((3, 16))
                iad[n, np.random.randint(16, size=1)] = 1
                self.data.append((iad, n))

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        input_dims = 3  # * 16
        classes = 3#7
        self.linear = nn.Sequential(
            nn.Linear(input_dims, classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view((-1, 3, 16))
        x, _ = torch.max(x, 2)
        # print(x)
        return self.linear(x)


if __name__ == "__main__":
    train_dataset = IADDataset(5)
    test_dataset = IADDataset(5)

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50)

    net = Model()
    params = list(net.parameters())
    net = nn.DataParallel(net, device_ids=[0]).cuda()
    net.train()

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.SGD(params,
                                0.05,
                                momentum=0.9,
                                weight_decay=0.005)
    epochs = 1000

    with torch.autograd.detect_anomaly():
        for e in range(epochs):
            cummulative_loss = 0
            for i, data_packet in enumerate(train_loader):
                data, label = data_packet[0], data_packet[1]

                data = data.float().cuda()
                label = label.cuda()

                logits = net(data)

                loss = criterion(logits, label)
                cummulative_loss += loss
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()


            print("loss:", cummulative_loss)

    correct = 0
    for i, data_packet in enumerate(test_loader):
        data, label = data_packet[0], data_packet[1]
        data = data.float().cuda()

        logits = net(data)
        predicted = logits.detach().cpu().numpy()
        predicted = np.argmax(predicted, axis=1)

        print("pred:", predicted, "exp:", label)
        print("logits:", logits)
