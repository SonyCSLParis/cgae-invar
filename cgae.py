"""
Created on April 13, 2018

@author: Stefan Lattner

Sony CSL Paris, France
Institute for Computational Perception, Johannes Kepler University, Linz
Austrian Research Institute for Artificial Intelligence, Vienna

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import to_numpy, cuda_tensor


class C_GAE(nn.Module):
    def __init__(self, kernel_size=(9, 120), factors=256, mapping=(128, 64)):
        """
        Convolutional Gated Autoencoder with two mapping layers for learning
        relative pitch representations from polyphonic sequence data.
        Note: For binary data, sigmoid non-linearity should be applied to
        the output (i.e. the reconstruction).

        :param context_length: temporal context (i.e. input kernel width)
        :param factors: number of factors
        :param mapping: 2-tuple, number of mapping units in first and second
        layer
        """
        super(C_GAE, self).__init__()
        assert kernel_size[0] % 2 == 1, "kernel_size[0] has to be an odd number"
        self.context_length = kernel_size[0]
        self.conv_x = nn.Conv2d(1, factors, kernel_size=kernel_size,
                                padding=(kernel_size[0]//2, 0), bias=False)
        self.bias_x = nn.Parameter(torch.zeros(kernel_size[1]))
        self.conv_y = nn.Conv2d(1, factors, kernel_size=(1, kernel_size[1]),
                                padding=(0, 0), bias=False)
        self.conv_m1 = nn.Conv2d(factors, mapping[0], kernel_size=(1, 1),
                                 padding=(0, 0), bias=True)
        self.conv_m2 = nn.Conv2d(mapping[0], mapping[1], kernel_size=(1, 1),
                                 padding=(0, 0), bias=True)

    def factors_(self, x, y):
        return self.conv_x(x) * self.conv_y(y)

    def mapping_(self, x, y):
        factors = self.factors_(x, y)
        m1 = F.tanh(self.conv_m1(factors))
        m2 = F.tanh(self.conv_m2(m1))
        return m2

    def recon_y_(self, m, x):
        decon_m2 = F.conv_transpose2d(m, self.conv_m2.weight,
                                      bias=self.conv_m1.bias)
        decon_m2 = F.conv_transpose2d(decon_m2, self.conv_m1.weight, bias=None)
        recon_y = F.conv_transpose2d(decon_m2 * self.conv_x(x),
                                     self.conv_y.weight, bias=None)
        return recon_y

    def forward(self, m, x):
        return self.recon_y_(m, x)

    def set_to_norm(self, val):
        """
        Sets the norms of all convolutional kernels of the C-GAE to a specific
        value.

        :param val: norms of kernels are set to this value
        """
        shape_x = self.conv_x.weight.size()
        conv_x_reshape = self.conv_x.weight.view(shape_x[0], -1)
        norms_x = ((conv_x_reshape ** 2).sum(1) ** .5).view(-1, 1)
        conv_x_reshape /= norms_x
        weight_x_new = to_numpy(conv_x_reshape.view(*shape_x)) * val
        self.conv_x.weight.data = cuda_tensor(weight_x_new)

        shape_y = self.conv_y.weight.size()
        conv_y_reshape = self.conv_y.weight.view(shape_y[0], -1)
        norms_y = ((conv_y_reshape ** 2).sum(1) ** .5).view(-1, 1)
        conv_y_reshape /= norms_y
        weight_y_new = to_numpy(conv_y_reshape.view(*shape_y)) * val
        self.conv_y.weight.data = cuda_tensor(weight_y_new)