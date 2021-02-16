import torch
import torch.nn as nn
import numpy as np

from torch_geometric.data import Data


class IAD2MaskedIAD(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, iad, threshold_values):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        ctx.save_for_backward(iad)

        # take IAD
        # nn.AdaptiveAvgPool1d(1)
        #threshold_values = []
        #locs = np.where(iad > threshold_values, 1)

        print(iad.shape, threshold_values.shape)
        print(type(iad), type(threshold_values))

        empty_locs = np.where(iad > threshold_values)#, 1)

        masked_iad = iad.clone()
        masked_iad[empty_locs] = 0

        #masked_idx = None

        return masked_iad

        # mask IAD
        #ctx.save_for_backward(input_tensor)
        #return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        iad, _ = ctx.saved_tensors
        grad_input = grad_output.clone()

        # grad_output should match the IAD shape
        grad_matrix = np.zeros_like(iad)
        grad_matrix[masked_idx] = 1

        grad_input *= grad_matrix
        return grad_input

        #input, = ctx.saved_tensors
        #grad_input = grad_output.clone()
        #grad_input[input < 0] = 0
        #return grad_input


        """The max pooling layer uses the maximum value out of all the ones in the kernel. So the gradient is 1 for the selected value and 0 for all the others.
So during the backward, the gradient of the output is (multiplied by 1 and) set to the selected value. All the others are set to 0.
To do so, it uses the indices returned by the forward pass."""


class MaskedIAD2ITR(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, iad):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(iad)

        # take masked IAD
        sparse_map = convert_iad_to_sparse_map(iad)

        # generate set of nodes
        # keep track of node weights to back-propagate the grad
        itr = convert_sparse_map_to_itr(sparse_map, iad)

        # generate edges and ITRs
        # ctx.save_for_backward(input)

        # return Graph
        return itr

        #ctx.save_for_backward(input)
        #return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        iad, _ = ctx.saved_tensors
        node_grad = grad_output.clone()

        grad_input = np.zeros_like(iad)

        count = 0
        for feature in sparse_map:
            for start, stop in sparse_map[feature]:
                grad_input[feature, start:stop] = node_grad[count]
                count += 1

        return grad_input

        #input, = ctx.saved_tensors
        #grad_input = grad_output.clone()
        #grad_input[input < 0] = 0
        #return grad_input


def convert_iad_to_sparse_map(iad, threshold_values, mask_idx):
    """Convert the IAD to a sparse map that denotes the start and stop times of each feature"""

    # apply threshold to get indexes where features are active
    locs = np.where(iad > threshold_values.reshape(len(mask_idx), 1))
    locs = np.dstack((locs[0], locs[1]))
    locs = locs[0]

    # get the start and stop times for each feature in the IAD
    if len(locs) != 0:
        sparse_map = []
        for i in range(iad.shape[0]):
            feature_row = locs[np.where(locs[:, 0] == i)][:, 1]

            # locate the start and stop times for the row of features
            start_stop_times = []
            if len(feature_row) != 0:
                start = feature_row[0]
                for j in range(1, len(feature_row)):
                    if feature_row[j - 1] + 1 < feature_row[j]:
                        start_stop_times.append([start, feature_row[j - 1] + 1])
                        start = feature_row[j]

                start_stop_times.append([start, feature_row[len(feature_row) - 1] + 1])

            # add start and stop times to sparse_map
            sparse_map.append(start_stop_times)
    else:
        sparse_map = [[]] * iad.shape[0]

    return sparse_map


def convert_sparse_map_to_itr(sparse_map, iad):

    relations = []
    events = []

    for f1 in range(len(sparse_map)):
        for e1 in range(len(sparse_map[f1])):
            e1_l = str(f1) + "_" + str(e1)
            e1_t = sparse_map[f1][e1]
            e1_weight = iad[f1, e1_t[0]:e1_t[1]].max()
            events.append((e1_l, e1_weight))

            for f2 in range(len(sparse_map)):
                for e2 in range(len(sparse_map[f2])):
                    e2_l = str(f2) + "_" + str(e2)
                    e2_t = sparse_map[f2][e2]

                    itr = find_relations(e1_t, e2_t)
                    if itr > 0 or (itr == 0 and f1 == f2):
                        relations.append((e1_l, e2_l, itr))

    # return relations
    e_map = {}

    node_x = np.zeros((len(events), len(sparse_map)))
    for e in range(len(events)):
        e_name = events[e][0]
        e_weight = events[e][1]

        e_map[e_name] = e
        node_x[e][int(e_name.split('_')[0])] = e_weight

    edge_index = []
    edge_attr = []
    for r in relations:
        e1, e2, itr = r
        edge_index.append((e_map[e1], e_map[e2]))
        edge_attr.append(itr)

    edge_index = np.array(edge_index).T
    edge_attr = np.array(edge_attr)

    return Data(node_x, edge_index=edge_index, edge_attr=edge_attr)


def find_relations(e1_t, e2_t):
    a1 = e1_t[0]
    a2 = e1_t[1]
    b1 = e2_t[0]
    b2 = e2_t[1]

    # before
    if a2 < b1:
        return 0  # 'b';

    # meets
    if a2 == b1:
        return 1  # 'm';

    # overlaps
    if a1 < b1 < a2 < b2:
        return 2  # 'o';

    # during
    if a1 < b1 and b2 < a2:
        return 3  # 'd';

    # finishes
    if b1 < a1 and a2 == b2:
        return 4  # 'f';

    # starts
    if a1 == b1 and a2 < b2:
        return 5  # 's';

    # equals
    if a1 == b1 and a2 == b2:
        return 6  # 'e';
    return -1


if __name__ == '__main__':
    import numpy as np

    f = np.load("/home/mbc2004/datasets/BlockConstruction/iad_i3d/train/rrr/rrr_0.npz")
    iad = f["data"]

    threshold_values = np.mean(iad, axis=1).reshape(-1, 1)
    print(iad.shape, threshold_values.shape)
    #locs = np.where(iad > threshold_values)
    #print(locs)

    masked_iad = IAD2MaskedIAD.apply(iad, threshold_values)
    itr = MaskedIAD2ITR.apply(masked_iad)
