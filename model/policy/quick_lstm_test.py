import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_obs = 8
        self.num_act = 4
        self.num_layers = 1
        self.hidden_size = 16

        self.lstm = nn.LSTM(input_size=self.num_obs+self.num_act, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_act)


    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=2, out=None)

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()

        state_y, (h_out, _) = self.lstm(x, (h_0.detach(), c_0.detach()))
        state_y = self.fc(state_y)
        x = state_y[:, -1, :]

        return x

class TraceDataset(Dataset):
    def __init__(self, mode="train"):
        super().__init__()
        self.data = np.load("/home/mbc2004/datasets/BlockConstruction/traces3.npy")
        self.num_obs = 8
        self.num_act = 4

        if mode:
            self.data = self.data[:90]
            all_data = []
            for obs, act in self.data:
                for i in range(1,len(obs)):
                    all_data.append((obs[:i], act[:i]))
            self.data = all_data

        else:
            self.data = self.data[90:]



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        obs, act = self.data[item]
        obs_o = np.zeros((len(obs), self.num_obs))
        act_o = np.zeros((len(act), self.num_act))

        for i in range(len(obs)):
            obs_o[i, obs[i]] = 1
            act_o[i, act[i]] = 1

        return obs_o, act_o


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
        c_loss = 0
        for e in range(10):

            cumulative_loss = 0

            for i, data_packet in enumerate(data_loader):

                obs, act = data_packet
                print("obs:", obs.shape, obs.dtype)
                #print(obs)
                print("act:", act.shape, act.dtype)
                #print(act)

                obs = obs[:, -1:]
                act = act[:, -1:]

                print(obs)
                print(act)

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
                c_loss += loss
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()
        print("loss:", c_loss)

    return model




def evaluate_action_trace(model, mode="evaluation"):
    dataset = TraceDataset(mode)
    data_loader = create_dataloader(dataset, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    net.eval()

    exp_list = []
    pred_list = []

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

                '''
                print("a:", a.shape)
                print("a_history:", a_history.shape)
                '''

                # compute output
                logits = net(o.float(), a_history.float())

                # get label information
                expected_label = label.cpu().detach().numpy()
                predicted_label = np.argmax(logits.cpu().detach().numpy(), axis=1)

                predicted_action_history.append(predicted_label)

            pred = np.array(predicted_action_history).reshape(1, -1)
            exp = torch.argmax(act, dim=2).detach().cpu().numpy()

            exp_list.extend(exp[0])
            pred_list.extend(pred[0])

            print("obs :", torch.argmax(obs, dim=2).detach().cpu().numpy())
            print("act :", exp)
            print("pred:", pred)
            print('')


    print("accuracy:", accuracy_score(y_true=exp, y_pred=pred))



if __name__ == '__main__':
    model = Model()

    train(model)
    evaluate_action_trace(model, mode="train")
    evaluate_action_trace(model)

