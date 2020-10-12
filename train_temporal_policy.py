"""
This code is for the training of a network using only the backbone model
"""
import torch
import numpy as np


def train_model(lfd_params, model, debug=True):

    # Create DataLoaders
    train_loader = lfd_params.create_dataloader(lfd_params, "train")

    # put model on GPU
    params = list(model.parameters())
    net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
    net.train()

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.SGD(params,
                                lfd_params.args.lr,
                                momentum=lfd_params.args.momentum,
                                weight_decay=lfd_params.args.weight_decay)

    # Train Network
    loss_record = []
    with torch.autograd.detect_anomaly():

        epoch = lfd_params.args.epochs
        for e in range(epoch):
            cummulative_loss = 0
            for i, data_packet in enumerate(train_loader):

                obs, state, action = data_packet

                # input shapes
                if debug and e == 0 and i == 0:
                    print("obs_x: ", obs.shape)
                    print("state_x: ", state.shape)

                # compute output
                action_logits = net(obs, state)

                # get loss
                loss = criterion(action_logits, action.cuda())
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()

                if debug and i % 100 == 0:
                    print("epoch: {:3d}/{:3d},  iter: {:6d}/{:6d}".format(e, epoch, i, len(train_loader)))
                    print("loss:", loss.cpu().detach().numpy())
                    print("expected:", action.cpu().detach().numpy())
                    print("pred:", np.argmax(action_logits.cpu().detach().numpy(), axis=1))
                    print("output:", action_logits.cpu().detach().numpy())

                cummulative_loss += loss.cpu().detach().numpy()
            loss_record.append(cummulative_loss)

    # save trained model parameters
    out_filename = lfd_params.generate_modelname()
    model.save_model()

    # show loss over time
    import matplotlib.pyplot as plt
    plt.plot(loss_record)
    plt.savefig("analysis/fig/loss_record.png")

    return out_filename


if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()
    lfd_params_obj.use_itrs()

    from model.model import LfDNetwork
    model_obj = LfDNetwork(lfd_params_obj, is_training=True)

    train_model(lfd_params_obj, model_obj)

