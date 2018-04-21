#!/usr/bin/env python

"""
Created on April 13, 2018

@author: Stefan Lattner

Sony CSL Paris, France
Institute for Computational Perception, Johannes Kepler University, Linz
Austrian Research Institute for Artificial Intelligence, Vienna

"""

import argparse
from random import randint

import torch
import torch.nn.functional as F
import torch.optim as optim

from cgae import C_GAE
from cqt import CQT
from plot import *
from regularize import lee_loss, l2_loss
from util import cuda_variable

parser = argparse.ArgumentParser(description='Train a convolutional gated'
                                             'autoencoder on audio data in '
                                             'CQT representation to obtain '
                                             'locally transposition-invariant '
                                             'features.')
parser.add_argument('filelist', type=str, default="",
                    help='text file containing a list of audio files for'
                         'training')
parser.add_argument('run_keyword', type=str, default="experiment1",
                    help='keyword used for output path')

parser.add_argument('--context-length', type=int, default=9, metavar='L',
                    help='width of input window (default: 9)')
parser.add_argument('--n-factors', type=int, default=256, metavar='N',
                    help='size of factor layer (default: 256)')
parser.add_argument('--n-mapping', type=tuple, default=(128, 64), metavar='N',
                    help='size of mapping layers (default: (128, 64))')

parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--epochs', type=int, default=501, metavar='N',
                    help='number of epochs to train (default: 501)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='R',
                    help='learning rate (default: 0.001)')
parser.add_argument('--sparsity-reg', type=float, default=4e-6, metavar='S',
                    help='sparsity regularization on top-most mapping layer ('
                         'default: 4e-6)')
parser.add_argument('--weight-reg', type=float, default=0.0, metavar='S',
                    help='weight regularization on the input filters ('
                         'default: 0.0)')

parser.add_argument('--block-size', type=int, default=1024, metavar='N',
                    help='length of one instance in batch (default: 1024)')
parser.add_argument('--n-bins', type=int, default=120, metavar='B',
                    help='number of frequency bins for CQT (default: 120)')
parser.add_argument('--bins-per-oct', type=int, default=24, metavar='B',
                    help='number of frequency bins per octave for CQT ('
                         'default: 24)')
parser.add_argument('--fmin', type=float, default=65.4, metavar='F',
                    help='minimum frequency for CQT (default: 65.4)')
parser.add_argument('--hop_length', type=int, default=448, metavar='L',
                    help='hop length for CQT (default: 448)')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--plot-interval', type=int, default=50, metavar='I',
                    help='how many epochs to wait before plotting network '
                         'status (default: 50)')
parser.add_argument('--refresh-cache', action="store_true", default=False,
                    help='reload and preprocess data')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

with open(args.filelist, 'r') as f:
    files = f.readlines()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

out_dir = os.path.join("output", args.run_keyword)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def transp(x, shift):
    """
    Transposes axis 3 (zero-based) of x by [shift] steps.
    Missing information is padded with zeros.

    :param x: the array to transpose
    :param shift: the transposition distance
    :return: x transposed
    """
    if shift == 0:
        return x

    pad = cuda_variable(
        torch.zeros(x.size(0), x.size(1), x.size(2), abs(shift)))

    if shift < 0:
        return torch.cat([pad, x[:, :, :, :-abs(shift)]], dim=3)
    return torch.cat([x[:, :, :, abs(shift):], pad], dim=3)


model = C_GAE(kernel_size=(args.context_length, args.n_bins),
              factors=args.n_factors, mapping=args.n_mapping)

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)


def get_trg_shift_dist():
    return 1 + model.context_length // 2


train_loader = torch.utils.data.DataLoader(
    CQT(files, trg_shift=get_trg_shift_dist(), block_size=args.block_size,
        n_bins=args.n_bins, bins_per_octave=args.bins_per_oct,
        fmin=args.fmin, hop_length=args.hop_length,
        convolutional=True, refresh_cache=args.refresh_cache),
    batch_size=args.batch_size, shuffle=True, **kwargs)


def train(epoch):
    """
    Trains the C-GAE for one epoch

    :param epoch: the current training epoch
    """
    model.train()
    losses = []
    for batch_idx, (x, y) in enumerate(train_loader):
        x = cuda_variable(x)
        y = cuda_variable(y)

        optimizer.zero_grad()

        # transpose input and target, in order to enforce
        # transposition-invariance
        shift = randint(-args.n_bins//2, args.n_bins//2)
        x = F.dropout(x, .5, training=True)
        x_trans = transp(x, shift)
        y_trans = transp(y, shift)

        # calculate mapping of untransposed data
        m = model.mapping_(x, y)
        # reconstruct transposed data using mapping of untransposed data
        y_recon = model(m, x_trans)

        # sparsity regularization on mapping
        lee_reg = lee_loss(m)
        # weight decay
        l2_reg = l2_loss(model.conv_x.weight) + l2_loss(model.conv_y.weight)

        # regularization strengths
        lee_fact = args.sparsity_reg
        l2_fact = args.weight_reg

        loss = F.mse_loss(y_recon, y_trans) + \
               lee_reg * lee_fact + l2_reg * l2_fact

        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])

        # restrict to (small) common norm
        model.set_to_norm(0.4)

    return losses


# Train and plot intermediate results

for epoch in range(1, args.epochs + 1):
    losses = train(epoch)
    print('Finished Epoch: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(
          epoch, args.epochs+1, 100. * epoch / args.epochs+1, np.mean(losses)))
    if epoch % args.plot_interval == 1:
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx == 0:
                x = cuda_variable(x)
                y = cuda_variable(y)
                plot_data(y, epoch, out_dir)
                batch = train_loader.dataset.__getitem__(0)
                plot_recon(model, x, y, epoch, out_dir)
                plot_mapping(model, x, y, epoch, out_dir)
                plot_kernels(model, epoch, out_dir)
                plot_histograms(model, x, y, epoch, out_dir)
            else:
                break

# Save the model
torch.save(model, os.path.join(out_dir, 'model.save'))
