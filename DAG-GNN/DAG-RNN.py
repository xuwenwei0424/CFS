
from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

from torch.autograd import Variable

def get_args():

    parser = argparse.ArgumentParser()

    # -----------data parameters ------
    # configurations
    parser.add_argument('--data_type', type=str, default= 'synthetic',
                        choices=['synthetic', 'discrete', 'real'],
                        help='choosing which experiment to do.')
    parser.add_argument('--data_filename', type=str, default= 'alarm',
                        help='data file name containing the discrete files.')
    parser.add_argument('--data_dir', type=str, default= 'data/',
                        help='data file name containing the discrete files.')
    parser.add_argument('--data_sample_size', type=int, default=5000,
                        help='the number of samples of data')
    parser.add_argument('--data_variable_size', type=int, default=10,
                        help='the number of variables in synthetic generated data')
    parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                        help='the type of DAG graph by generation method')
    parser.add_argument('--graph_degree', type=int, default=2,
                        help='the number of degree in generated DAG graph')
    parser.add_argument('--graph_sem_type', type=str, default='linear-gauss',
                        help='the structure equation model (SEM) parameter type')
    parser.add_argument('--graph_linear_type', type=str, default='nonlinear_2',
                        help='the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z, nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z')
    parser.add_argument('--edge-types', type=int, default=2,
                        help='The number of edge types to infer.')
    parser.add_argument('--x_dims', type=int, default=1, #changed here
                        help='The number of input dimensions: default 1.')
    parser.add_argument('--z_dims', type=int, default=1,
                        help='The number of latent variable dimensions: default the same as variable size.')

    # -----------training hyperparameters
    parser.add_argument('--optimizer', type = str, default = 'Adam',
                        help = 'the choice of optimizer used')
    parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                        help = 'threshold for learned adjacency matrix binarization')
    parser.add_argument('--tau_A', type = float, default=0.0,
                        help='coefficient for L-1 norm of A.')
    parser.add_argument('--lambda_A',  type = float, default= 0.,
                        help='coefficient for DAG constraint h(A).')
    parser.add_argument('--c_A',  type = float, default= 1,
                        help='coefficient for absolute value h(A).')
    parser.add_argument('--use_A_connect_loss',  type = int, default= 0,
                        help='flag to use A connect loss')
    parser.add_argument('--use_A_positiver_loss', type = int, default = 0,
                        help = 'flag to enforce A must have positive values')


    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default= 300,
                        help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default = 100, # note: should be divisible by sample size, otherwise throw an error
                        help='Number of samples per batch.')
    parser.add_argument('--lr', type=float, default=3e-3,  # basline rate = 1e-3
                        help='Initial learning rate.')
    parser.add_argument('--encoder-hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--decoder-hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature for Gumbel softmax.')
    parser.add_argument('--k_max_iter', type = int, default = 1e2,
                        help ='the max iteration number for searching lambda and c')
    parser.add_argument('--encoder', type=str, default='mlp',
                        help='Type of path encoder model (mlp, or sem).')
    parser.add_argument('--decoder', type=str, default='mlp',
                        help='Type of decoder model (mlp, or sim).')
    parser.add_argument('--no-factor', action='store_true', default=False,
                        help='Disables factor graph model.')
    parser.add_argument('--suffix', type=str, default='_springs5',
                        help='Suffix for training data (e.g. "_charged".')
    parser.add_argument('--encoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--decoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--save-folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    parser.add_argument('--load-folder', type=str, default='',
                        help='Where to load the trained model if finetunning. ' +
                            'Leave empty to train from scratch')


    parser.add_argument('--h_tol', type=float, default = 1e-8,
                        help='the tolerance of error of h(A) to zero')
    parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                        help='Num steps to predict before re-using teacher forcing.')
    parser.add_argument('--lr-decay', type=int, default=200,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default= 1.0,
                        help='LR decay factor.')
    parser.add_argument('--skip-first', action='store_true', default=False,
                        help='Skip first edge type in decoder, i.e. it represents no-edge.')
    parser.add_argument('--var', type=float, default=5e-5,
                        help='Output variance.')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='Uses discrete samples in training forward pass.')
    parser.add_argument('--prior', action='store_true', default=False,
                        help='Whether to use sparsity prior.')
    parser.add_argument('--dynamic-graph', action='store_true', default=False,
                        help='Whether test with dynamically re-computed graph.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.factor = not args.no_factor

    return args

class MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa

class MLPDEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(MLPDEncoder, self).__init__()

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor

        self.Wa = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.fc1 = nn.Linear(n_hid, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)

        n_var = adj_A.shape[0]
        self.embed = nn.Embedding(n_out, n_hid)
        self.dropout_prob = do_prob
        self.alpha =  nn.Parameter(Variable(torch.div(torch.ones(n_var, n_out),n_out)).double(), requires_grad = True)
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        adj_A1 = torch.sinh(3.*self.adj_A)

        adj_Aforz = preprocess_adj_new(adj_A1)
        adj_A = torch.eye(adj_A1.size()[0]).double()

        bninput = self.embed(inputs.long().view(-1, inputs.size(2)))
        bninput = bninput.view(*inputs.size(),-1).squeeze()
        H1 = F.relu((self.fc1(bninput)))
        x = (self.fc2(H1))


        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa

        prob = my_softmax(logits, -1)
        alpha = my_softmax(self.alpha, -1)

        return x, prob, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa, alpha


class SEMEncoder(nn.Module):
    """SEM encoder module."""
    def __init__(self, n_in, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(SEMEncoder, self).__init__()

        self.factor = factor
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad = True))
        self.dropout_prob = do_prob
        self.batch_size = batch_size

    def init_weights(self):
        nn.init.xavier_normal(self.adj_A.data)

    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_A = I-A^T, adj_A_inv = (I-A^T)^(-1)
        adj_A = preprocess_adj_new((adj_A1))
        adj_A_inv = preprocess_adj_new1((adj_A1))

        meanF = torch.matmul(adj_A_inv, torch.mean(torch.matmul(adj_A, inputs), 0))
        logits = torch.matmul(adj_A, inputs-meanF)

        return inputs-meanF, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A


#[YY] delete it?
class MLPDDecoder(nn.Module):
    """MLP decoder module. OLD DON"T USE
    """

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDDecoder, self).__init__()

        self.bn0 = nn.BatchNorm1d(n_in_node * 1, affine=True)
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_hid, bias = True)
        self.out_fc3 = nn.Linear(n_hid, n_out, bias = True)
#        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        self.bn1 = nn.BatchNorm1d(n_in_node * 1, affine=True)
#         self.W3 = Variable(torch.from_numpy(W3).float())
#         self.W4 = Variable(torch.from_numpy(W4).float())

        # TODO check if this is indeed correct
        #self.adj_A = encoder.adj_A
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):

        # # copy adj_A batch size
        # adj_A = self.adj_A.unsqueeze(0).repeat(self.batch_size, 1, 1)

        adj_A_new = torch.eye(origin_A.size()[0]).double()#preprocess_adj(origin_A)#
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa #.unsqueeze(2) #.squeeze(1).unsqueeze(1).repeat(1, self.data_variable_size, 1) # torch.repeat(torch.transpose(input_z), torch.ones(n_in_node), axis=0)

        adj_As = adj_A_new

        #mat_z_max = torch.matmul(adj_A_new, my_normalize(mat_z))

#        mat_z_max = (torch.max(mat_z, torch.matmul(adj_As, mat_z)))
        H3 = F.relu(self.out_fc1((mat_z)))

        #H3_max = torch.matmul(adj_A_new, my_normalize(H3))
#        H3_max = torch.max(H3, torch.matmul(adj_As, H3))

#        H4 = F.relu(self.out_fc2(H3))

        #H4_max = torch.matmul(adj_A_new, my_normalize(H4))
#        H4_max = torch.max(H4, torch.matmul(adj_As, H4))

#        H5 = F.relu(self.out_fc4(H4_max)) + H3

        #H5_max = torch.max(H5, torch.matmul(adj_As, H5))

        # mu and sigma
        out = self.out_fc3(H3)

        return mat_z, out, adj_A_tilt#, self.adj_A

#[YY] delete it?
class MLPDiscreteDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDiscreteDecoder, self).__init__()
#        self.msg_fc1 = nn.ModuleList(
#            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
#        self.msg_fc2 = nn.ModuleList(
#            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
#        self.msg_out_shape = msg_out
#        self.skip_first_edge_type = skip_first

        self.bn0 = nn.BatchNorm1d(n_in_node * 1, affine=True)
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_hid, bias = True)
#        self.out_fc4 = nn.Linear(n_hid, n_hid, bias=True)
        self.out_fc3 = nn.Linear(n_hid, n_out, bias = True)
#        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        self.bn1 = nn.BatchNorm1d(n_in_node * 1, affine=True)
#         self.W3 = Variable(torch.from_numpy(W3).float())
#         self.W4 = Variable(torch.from_numpy(W4).float())

        # TODO check if this is indeed correct
        #self.adj_A = encoder.adj_A
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size
        self.softmax = nn.Softmax(dim=2)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):

        # # copy adj_A batch size
        # adj_A = self.adj_A.unsqueeze(0).repeat(self.batch_size, 1, 1)

        adj_A_new = torch.eye(origin_A.size()[0]).double()#preprocess_adj(origin_A)#
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa #.unsqueeze(2) #.squeeze(1).unsqueeze(1).repeat(1, self.data_variable_size, 1) # torch.repeat(torch.transpose(input_z), torch.ones(n_in_node), axis=0)

        adj_As = adj_A_new

        #mat_z_max = torch.matmul(adj_A_new, my_normalize(mat_z))

#        mat_z_max = (torch.max(mat_z, torch.matmul(adj_As, mat_z)))
        H3 = F.relu(self.out_fc1((mat_z)))

        #H3_max = torch.matmul(adj_A_new, my_normalize(H3))
#        H3_max = torch.max(H3, torch.matmul(adj_As, H3))

#        H4 = F.relu(self.out_fc2(H3))

        #H4_max = torch.matmul(adj_A_new, my_normalize(H4))
#        H4_max = torch.max(H4, torch.matmul(adj_As, H4))

#        H5 = F.relu(self.out_fc4(H4_max)) + H3

        #H5_max = torch.max(H5, torch.matmul(adj_As, H5))

        # mu and sigma
        out = self.softmax(self.out_fc3(H3)) # discretized log

        return mat_z, out, adj_A_tilt#, self.adj_A


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt

class SEMDecoder(nn.Module):
    """SEM decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(SEMDecoder, self).__init__()

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):

        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa)
        out = mat_z

        return mat_z, out-Wa, adj_A_tilt

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

def stau(w, tau):
    prox_plus = torch.nn.Threshold(0.,0.)
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1


def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.linalg as slin
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
import glob
import re
import pickle
import math
from torch.optim.adam import Adam

# data generating functions

def simulate_random_dag(d: int,
                        degree: float,
                        graph_type: str,
                        w_range: tuple = (0.5, 2.0)) -> nx.DiGraph:
    """Simulate random DAG with some expected degree.

    Args:
        d: number of nodes
        degree: expected node degree, in + out
        graph_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)

    Returns:
        G: weighted DAG
    """
    if graph_type == 'erdos-renyi':
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    elif graph_type == 'barabasi-albert':
        m = int(round(degree / 2))
        B = np.zeros([d, d])
        bag = [0]
        for ii in range(1, d):
            dest = np.random.choice(bag, size=m)
            for jj in dest:
                B[ii, jj] = 1
            bag.append(ii)
            bag.extend(dest)
    elif graph_type == 'full':  # ignore degree, only for experimental use
        B = np.tril(np.ones([d, d]), k=-1)
    else:
        raise ValueError('unknown graph type')
    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G


def simulate_sem(G: nx.DiGraph,
                 n: int, x_dims: int,
                 sem_type: str,
                 linear_type: str,
                 noise_scale: float = 1.0) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM

    Returns:
        X: [n,d] sample matrix
    """
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros([n, d, x_dims])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        if linear_type == 'linear':
            eta = X[:, parents, 0].dot(W[parents, j])
        elif linear_type == 'nonlinear_1':
            eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
        elif linear_type == 'nonlinear_2':
            eta = (X[:, parents, 0]+0.5).dot(W[parents, j])
        else:
            raise ValueError('unknown linear data type')

        if sem_type == 'linear-gauss':
            if linear_type == 'linear':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_1':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_2':
                X[:, j, 0] = 2.*np.sin(eta) + eta + np.random.normal(scale=noise_scale, size=n)
        elif sem_type == 'linear-exp':
            X[:, j, 0] = eta + np.random.exponential(scale=noise_scale, size=n)
        elif sem_type == 'linear-gumbel':
            X[:, j, 0] = eta + np.random.gumbel(scale=noise_scale, size=n)
        else:
            raise ValueError('unknown sem type')
    if x_dims > 1 :
        for i in range(x_dims-1):
            X[:, :, i+1] = np.random.normal(scale=noise_scale, size=1)*X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
        X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
    return X


def simulate_population_sample(W: np.ndarray,
                               Omega: np.ndarray) -> np.ndarray:
    """Simulate data matrix X that matches population least squares.

    Args:
        W: [d,d] adjacency matrix
        Omega: [d,d] noise covariance matrix

    Returns:
        X: [d,d] sample matrix
    """
    d = W.shape[0]
    X = np.sqrt(d) * slin.sqrtm(Omega).dot(np.linalg.pinv(np.eye(d) - W))
    return X


def count_accuracy(G_true: nx.DiGraph,
                   G: nx.DiGraph,
                   G_und: nx.DiGraph = None) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        G_true: ground truth graph
        G: predicted graph
        G_und: predicted undirected edges in CPDAG, asymmetric

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    B_und = None if G_und is None else nx.to_numpy_array(G_und)
    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size



#========================================
# VAE utility functions
#========================================
def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise).double()
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def gauss_sample_z(logits,zsize):
    U = torch.randn(logits.size(0),zsize).double()
    x = torch.zeros(logits.size(0),1, zsize).double()
    for j in range(logits.size(0)):
        x[j,0,:] = U[j,:]*torch.exp(logits[j,0,zsize:2*zsize])+logits[j,0,0:zsize]
    return x

def gauss_sample_z_new(logits,zsize):
    U = torch.randn(logits.size(0),logits.size(1),zsize).double()
    x = torch.zeros(logits.size(0),logits.size(1),zsize).double()
    x[:, :, :] = U[:, :, :] + logits[:, :, 0:zsize]
    return x

def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('_graph' + extension))


def read_BNrep(args):
    '''load results from BN repository'''

    if args.data_filename == 'alarm':
        data_dir = os.path.join(args.data_dir, 'alarm/')
    elif args.data_filename == 'child':
        data_dir = os.path.join(args.data_dir, 'child/')
    elif args.data_filename =='hail':
        data_dir = os.path.join(args.data_dir, 'hail/')
    elif args.data_filename =='alarm10':
        data_dir = os.path.join(args.data_dir, 'alarm10/')
    elif args.data_filename == 'child10':
        data_dir = os.path.join(args.data_dir, 'child10/')
    elif args.data_filename == 'pigs':
        data_dir = os.path.join(args.data_dir, 'pigs/')

    all_data = dict()
    # read text files
    file_pattern = data_dir +"*_s*_v*.txt"
    all_files = glob.iglob(file_pattern)
    for file in all_files:
        match = re.search('/([\w]+)_s([\w]+)_v([\w]+).txt', file)
        dataset, samplesN, version = match.group(1), match.group(2),match.group(3)

        # read file
        data = np.loadtxt(file, skiprows =0, dtype=np.int32)
        if samplesN not in all_data:
            all_data[samplesN] = dict()

        all_data[samplesN][version] = data

    # read ground truth graph
    from os import listdir

    file_pattern = data_dir + "*_graph.txt"
    files = glob.iglob(file_pattern)
    for f in files:
        graph = np.loadtxt(f, skiprows =0, dtype=np.int32)

    return all_data, graph # in dictionary

def load_data_discrete(args, batch_size=1000, suffix='', debug = False):
    #  # configurations
    n, d = args.data_sample_size, args.data_variable_size
    graph_type, degree, sem_type = args.graph_type, args.graph_degree, args.graph_sem_type

    if args.data_type == 'synthetic':
        # generate data
        G = simulate_random_dag(d, degree, graph_type)
        X = simulate_sem(G, n, sem_type)

    elif args.data_type == 'discrete':
        # get benchmark discrete data
        if args.data_filename.endswith('.pkl'):
            with open(os.path.join(args.data_dir, args.data_filename), 'rb') as handle:
                X = pickle.load(handle)
        else:
            all_data, graph = read_BNrep(args)
            G = nx.DiGraph(graph)
            X = all_data['1000']['1']

    max_X_card = np.amax(X) + 1

    feat_train = torch.FloatTensor(X)
    feat_valid = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)

    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, G, max_X_card, X

def load_data(args, batch_size=1000, suffix='', debug = False):
    #  # configurations
    n, d = args.data_sample_size, args.data_variable_size
    graph_type, degree, sem_type, linear_type = args.graph_type, args.graph_degree, args.graph_sem_type, args.graph_linear_type
    x_dims = args.x_dims

    if args.data_type == 'synthetic':
        # generate data
        G = simulate_random_dag(d, degree, graph_type)
        X = simulate_sem(G, n, x_dims, sem_type, linear_type)

    elif args.data_type == 'discrete':
        # get benchmark discrete data
        if args.data_filename.endswith('.pkl'):
            with open(os.path.join(args.data_dir, args.data_filename), 'rb') as handle:
                X = pickle.load(handle)
        else:
            all_data, graph = read_BNrep(args)
            G = nx.DiGraph(graph)
            X = all_data['1000']['1']


    feat_train = torch.FloatTensor(X)
    feat_valid = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)

    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, G


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
           data_norm.transpose(2, 3) - \
           2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                             dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket



def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - torch.log(log_prior + eps))
    return kl_div.sum() / (num_atoms)

def kl_gaussian(preds, zsize):
    predsnew = preds.squeeze(1)
    mu = predsnew[:,0:zsize]
    log_sigma = predsnew[:,zsize:2*zsize]
    kl_div = torch.exp(2*log_sigma) - 2*log_sigma + mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)) - zsize)*0.5

def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5

def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

def nll_catogrical(preds, target, add_const = False):
    '''compute the loglikelihood of discrete variables
    '''
    # loss = nn.CrossEntropyLoss()

    total_loss = 0
    for node_size in range(preds.size(1)):
        total_loss += - (torch.log(preds[:,node_size, target[:, node_size].long()]) * target[:, node_size]).mean()

    return total_loss

def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))

#Symmetrically normalize adjacency matrix.
def normalize_adj(adj):
    rowsum = torch.abs(torch.sum(adj,1))
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    myr = torch.matmul(torch.matmul(d_mat_inv_sqrt,adj),d_mat_inv_sqrt)
    myr[isnan(myr)] = 0.
    return myr

def preprocess_adj(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() + (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized

def isnan(x):
    return x!=x

def my_normalize(z):
    znor = torch.zeros(z.size()).double()
    for i in range(z.size(0)):
        testnorm = torch.norm(z[i,:,:], dim=0)
        znor[i,:,:] = z[i,:,:]/testnorm
    znor[isnan(znor)] = 0.0
    return znor

def sparse_to_tuple(sparse_mx):
#    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def matrix_poly(matrix, d):
    x = torch.eye(d).double()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)


# matrix loss: makes sure at least A connected to another parents for child
def A_connect_loss(A, tol, z):
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss +=  2 * tol - torch.sum(torch.abs(A[:,i])) - torch.sum(torch.abs(A[i,:])) + z * z
    return loss

# element loss: make sure each A_ij > 0
def A_positive_loss(A, z_positive):
    result = - A + z_positive * z_positive
    loss =  torch.sum(result)

    return loss


'''
COMPUTE SCORES FOR BN
'''
def compute_BiCScore(G, D):
    '''compute the bic score'''
    # score = gm.estimators.BicScore(self.data).score(self.model)
    origin_score = []
    num_var = G.shape[0]
    for i in range(num_var):
        parents = np.where(G[:,i] !=0)
        score_one = compute_local_BiCScore(D, i, parents)
        origin_score.append(score_one)

    score = sum(origin_score)

    return score


def compute_local_BiCScore(np_data, target, parents):
    # use dictionary
    sample_size = np_data.shape[0]
    var_size = np_data.shape[1]

    # build dictionary and populate
    count_d = dict()
    if len(parents) < 1:
        a = 1

    # unique_rows = np.unique(self.np_data, axis=0)
    # for data_ind in range(unique_rows.shape[0]):
    #     parent_combination = tuple(unique_rows[data_ind,:].reshape(1,-1)[0])
    #     count_d[parent_combination] = dict()
    #
    #     # build children
    #     self_value = tuple(self.np_data[data_ind, target].reshape(1,-1)[0])
    #     if parent_combination in count_d:
    #         if self_value in count_d[parent_combination]:
    #             count_d[parent_combination][self_value] += 1.0
    #         else:
    #             count_d[parent_combination][self_value] = 1.0
    #     else:
    #         count_d[parent_combination] = dict()
    #         count_d

    # slower implementation
    for data_ind in range(sample_size):
        parent_combination = tuple(np_data[data_ind, parents].reshape(1, -1)[0])
        self_value = tuple(np_data[data_ind, target].reshape(1, -1)[0])
        if parent_combination in count_d:
            if self_value in count_d[parent_combination]:
                count_d[parent_combination][self_value] += 1.0
            else:
                count_d[parent_combination][self_value] = 1.0
        else:
            count_d[parent_combination] = dict()
            count_d[parent_combination][self_value] = 1.0

    # compute likelihood
    loglik = 0.0
    # for data_ind in range(sample_size):
    # if len(parents) > 0:
    num_parent_state = np.prod(np.amax(np_data[:, parents], axis=0) + 1)
    # else:
    #    num_parent_state = 0
    num_self_state = np.amax(np_data[:, target], axis=0) + 1

    for parents_state in count_d:
        local_count = sum(count_d[parents_state].values())
        for self_state in count_d[parents_state]:
            loglik += count_d[parents_state][self_state] * (
                        math.log(count_d[parents_state][self_state] + 0.1) - math.log(local_count))

    # penality
    num_param = num_parent_state * (
                num_self_state - 1)  # count_faster(count_d) - len(count_d) - 1 # minus top level and minus one
    bic = loglik - 0.5 * math.log(sample_size) * num_param

    return bic

#===================================
# training:
#===================================
def train(args, train_loader, epoch, encoder, decoder, scheduler, best_val_loss, dataset, ground_truth_G, lambda_A, c_A, optimizer, rel_rec, rel_send):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_train = []

    encoder.train()
    decoder.train()
    scheduler.step()


    # update optimizer
    optimizer, lr = update_optimizer(optimizer, args.lr, c_A)
    for batch_idx, (data, relations) in enumerate(train_loader):
        data = dataset
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data).double(), Variable(relations).double()

        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()

        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send)  # logits is of size: [num_sims, z_dims]
        edges = logits

        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, args.data_variable_size * args.x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')

        target = data
        preds = output
        variance = 0.

        # reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_A = origin_A # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))

        # other loss term
        if args.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if args.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        # compute h(A)
        h_A = _h_A(origin_A, args.data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss #+  0.01 * torch.sum(variance * variance)


        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, args.tau_A*lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))


        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_train.append(shd)

        np.savetxt("predG_DAG_Sixcaco2_fs100_new_1_5.csv", graph, delimiter=",")
        # num_nodes = len(graph[0])
        # res1 = set()
        # res2 = set()
        # res3 = set()
        # for i in range(num_nodes):
        #     if data[0][i] != 0 and i != 0:
        #         res1.add(i)
        #         print("孩子：0-->{}".format(i))
        #         for j in range(num_nodes):
        #             if data[j][i] != 0 and j != 0:
        #                 res3.add(j)
        #                 print("配偶：{}-->{}".format(j, i))
        #
        #     if data[i][0] != 0 and i != 0:
        #         res2.add(i)
        #         print("父亲：{}-->0".format(i))
        #
        # print("孩子节点的总数是{}，分别为：{}".format(len(res1), res1))
        # print("父亲节点的总数是{}，分别为：{}".format(len(res2), res2))
        # print("配偶节点的总数是{}，分别为：{}".format(len(res3), res3))
        # print("总数是{}".format(len(res1) + len(res2) + len(res3)))

    print(h_A.item())
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'shd_train: {:.10f}'.format(np.mean(shd_train)),
          'time: {:.4f}s'.format(time.time() - t))

    if 'graph' not in vars():
        print('error on assign')


    return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A

#===================================
# main
#===================================
def DAG_GNN_main(dataset):

    args = get_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    args.data_sample_size = dataset.shape[0]
    args.data_variable_size = dataset.shape[1]
    # ================================================
    # get data: experiments = {synthetic SEM, ALARM}
    # ================================================
    train_loader, valid_loader, test_loader, ground_truth_G = load_data(args, args.batch_size, args.suffix)
    dataset = torch.from_numpy(dataset)
    dataset = dataset.unsqueeze(-1)
    #===================================
    # load modules
    #===================================
    # add adjacency matrix A
    num_nodes = args.data_variable_size
    adj_A = np.zeros((num_nodes, num_nodes))

    if args.encoder == 'mlp':
        encoder = MLPEncoder(args.data_variable_size * args.x_dims, args.x_dims, args.encoder_hidden,
                            int(args.z_dims), adj_A,
                            batch_size = args.batch_size,
                            do_prob = args.encoder_dropout, factor = args.factor).double()
    elif args.encoder == 'sem':
        encoder = SEMEncoder(args.data_variable_size * args.x_dims, args.encoder_hidden,
                            int(args.z_dims), adj_A,
                            batch_size = args.batch_size,
                            do_prob = args.encoder_dropout, factor = args.factor).double()

    if args.decoder == 'mlp':
        decoder = MLPDecoder(args.data_variable_size * args.x_dims,
                            args.z_dims, args.x_dims, encoder,
                            data_variable_size = args.data_variable_size,
                            batch_size = args.batch_size,
                            n_hid=args.decoder_hidden,
                            do_prob=args.decoder_dropout).double()
    elif args.decoder == 'sem':
        decoder = SEMDecoder(args.data_variable_size * args.x_dims,
                            args.z_dims, 2, encoder,
                            data_variable_size = args.data_variable_size,
                            batch_size = args.batch_size,
                            n_hid=args.decoder_hidden,
                            do_prob=args.decoder_dropout).double()

    # Generate off-diagonal interaction graph
    off_diag = np.ones([args.data_variable_size, args.data_variable_size]) - np.eye(args.data_variable_size)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    #===================================
    # set up training parameters
    #===================================
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=args.lr)
    elif args.optimizer == 'LBFGS':
        optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                            lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                            lr=args.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                    gamma=args.gamma)

    # Linear indices of an upper traingular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(args.data_variable_size)
    tril_indices = get_tril_offdiag_indices(args.data_variable_size)

    if args.cuda:
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    t_total = time.time()
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = args.c_A
    lambda_A = args.lambda_A
    h_A_new = torch.tensor(1.)
    h_tol = args.h_tol
    k_max_iter = int(args.k_max_iter)
    h_A_old = np.inf

    try:
        for step_k in range(k_max_iter):
            while c_A < 1e+20:
                for epoch in range(args.epochs):
                    ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(args, train_loader, epoch, encoder, decoder, scheduler, best_ELBO_loss, dataset, ground_truth_G, lambda_A, c_A, optimizer, rel_rec, rel_send)
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph

                print("Optimization Finished!")
                print("Best Epoch: {:04d}".format(best_epoch))
                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # update parameters
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, args.data_variable_size)
                if h_A_new.item() > 0.25 * h_A_old:
                    c_A*=10
                else:
                    break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_new.item() <= h_tol:
                break

        # test()
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0
        return graph

    except KeyboardInterrupt:
        # print the best anway
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0
        return graph

# main
dataset = np.loadtxt(open("dataset", "rb"), delimiter=",")
dataset -= np.mean(dataset,axis=0)
DAG_GNN_main(dataset)