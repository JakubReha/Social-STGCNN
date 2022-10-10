import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim


class ConvGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True):
        super(ConvGraphical, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=bias)

    def forward(self, x, A):
        x = self.conv(x)
        x = torch.einsum('ncv,nvw->ncw', (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()


        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        self.use_mdn = use_mdn

        self.gcn = ConvGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                1,
                1,
                0,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1),
                nn.BatchNorm1d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class social_stgcnn(nn.Module):
    def __init__(self, n_stgcnn=5, n_txpcnn=1, input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3):
        super(social_stgcnn, self).__init__()
        self.pred_seq_len = pred_seq_len
        self.n_stgcnn = n_stgcnn
        self.output_feat = output_feat
        self.input_feat = input_feat
        self.st_gcns = nn.ModuleList()
        self.st_rnns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        self.st_rnns.append(torch.nn.RNN(output_feat, output_feat, batch_first=True))
        self.extrapolate = nn.Linear(output_feat*(self.n_stgcnn), pred_seq_len*output_feat)
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))
            self.st_rnns.append(torch.nn.RNN(output_feat, output_feat, batch_first=True, nonlinearity='relu'))


    def forward(self, v, a):
        # batch, channels, time, in_nodes --> time, channels, in_nodes
        v = v.squeeze().contiguous()
        v = v.view(v.shape[1], v.shape[0], v.shape[2])
        h_all = []
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)
            # in_nodes, time, channels
            v = v.view(v.shape[2], v.shape[0], v.shape[1])
            v, h = self.st_rnns[k](v)
            h_all.append(h)
            # time, channels, in_nodes
            v = v.contiguous()
            v = v.view(v.shape[1], v.shape[2], v.shape[0])

        # layers, in_nodes, channels
        v = torch.cat(h_all)
        #Extrapolation
        v = self.extrapolate(v.reshape(v.shape[1], -1))
        # batch, channels, time, in_nodes
        v = v.reshape(-1, self.pred_seq_len, self.output_feat)
        v = v.view(v.shape[2], v.shape[1], v.shape[0])
        v = v.unsqueeze(0)
        return v, a
