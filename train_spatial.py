"""
This code is for the training of a network using only the backbone model
"""
import torch
import os


def train(lfd_params, model):

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
            for i, data_packet in enumerate(train_loader):

                obs, state, action = data_packet

                # input shapes
                if e == 0 and i == 0:
                    print("obs_x: ", obs.shape)
                    print("state_x: ", state.shape)

                # compute output
                action_logits = net(obs)

                # get loss
                loss = criterion(action_logits, action.cuda())
                loss.backward()

                # optimize SGD
                optimizer.step()
                optimizer.zero_grad()

                if i % 100 == 0:
                    print("epoch: {:3d}/{:3d},  iter: {:6d}/{:6d}".format(e, epoch, i, len(train_loader)))
                    print("loss:", loss.cpu().detach().numpy())
                    print("expected:", action.cpu().detach().numpy())
                    print("output:", action_logits.cpu().detach().numpy())

                loss_record.append(loss.cpu().detach().numpy())

    # save trained model parameters
    model.save_model()

    # show loss over time
    import matplotlib.pyplot as plt
    plt.plot(loss_record)
    fig_filename = os.path.join(lfd_params.args.log_dir, lfd_params.args.save_id, "train_loss.png")
    plt.savefig(fig_filename)


if __name__ == '__main__':

    from parameter_parser import parse_model_args
    lfd_params_obj = parse_model_args()

    from model.spatial_feature_extractor import SpatialFeatureExtractor
    model_obj = SpatialFeatureExtractor(lfd_params_obj, is_training=True)

    train(lfd_params_obj, model_obj)

