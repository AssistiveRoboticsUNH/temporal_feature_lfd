import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=8, hidden_size=32,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        x = self.lstm(x)
        x = self.fc(x)
        return x

class TraceDataset(Dataset):
    def __init__(self, mode="train"):
        super().__init__()
        self.data = np.load("/home/mbc2004/datasets/BlockConstruction/traces4.npy")

        if mode:
            self.data = self.data[:90]
        else:
            self.data = self.data[90:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        obs, act = self.data[item]
        return obs, act


def create_dataloader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True)


def train(model):

    dataset = TraceDataset()
    data_loader = create_dataloader(dataset, shuffle=True)

    # put model on GPU
    params = list(model.parameters())
    net = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    net.train()

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.Adam(params, lr=0.0001)

    # Train Network
    loss_record = []
    with torch.autograd.detect_anomaly():
        for e in range(50):

            cumulative_loss = 0

            for i, data_packet in enumerate(data_loader):

                obs, act = data_packet
                print("obs:", obs.shape, obs.dtype)
                print("act:", act.shape, act.dtype)

                # obtain label
                label = act[:, -1]
                label = torch.argmax(label, dim=1)

                # hide label
                act[:, -1] = 0

                # compute output

                logits = net(obs.float(), act.float())

                # get loss
                print("label:", label.shape, label.dtype)
                loss = criterion(logits, label.long().cuda())
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()

    return model




def evaluate_action_trace(model, mode="evaluation"):
    dataset = TraceDataset(mode)
    data_loader = create_dataloader(dataset, shuffle=True)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    net.eval()

    with torch.no_grad():
        for i, data_packet in enumerate(data_loader):
            obs, act = data_packet

            predicted_action_history = []

            for j in range(1, act.shape[1]+1):

                o = obs[:, :j]
                a = act[:, :j]

                # obtain label
                label = a[:, -1]
                label = torch.argmax(label, dim=1)

                # prepare a_history
                a_history = np.zeros((1, (len(predicted_action_history)+1), 4))
                for k in range(len(predicted_action_history)):
                    a_history[0, k, predicted_action_history[k]] = 1
                a_history = torch.from_numpy(a_history)

                print("a:", a.shape)
                print("a_history:", a_history.shape)

                # compute output
                logits = net(o.float(), a_history.float())

                # get label information
                expected_label = label.cpu().detach().numpy()
                predicted_label = np.argmax(logits.cpu().detach().numpy(), axis=1)

                predicted_action_history.append(predicted_label)

            print("act :", act)
            print("pred:", predicted_action_history)
            print('')




if __name__ == '__main__':
    model = Model()

    train(model)
    evaluate_action_trace(model)

