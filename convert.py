#!/usr/bin/env python

"""
Created on April 13, 2018

@author: Stefan Lattner

Sony CSL Paris, France
Institute for Computational Perception, Johannes Kepler University, Linz
Austrian Research Institute for Artificial Intelligence, Vienna

"""

import argparse

import sys
import torch
import torch.nn.functional as F

from cqt import CQT, save_pyc_bz
from plot import *
from util import cuda_variable

parser = argparse.ArgumentParser(description='Convert a list of audio files '
                                             'to interval representation '
                                             'using a trained model.')
parser.add_argument('filelist', type=str, default="",
                    help='text file containing a list of audio files for'
                         'conversion')
parser.add_argument('run_keyword', type=str, default="experiment1",
                    help='keyword used for input path')
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
parser.add_argument('--refresh-cache', action="store_true", default=False,
                    help='reload and preprocess data')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

with open(args.filelist, 'r') as f:
    files = f.readlines()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

out_dir = os.path.join("output", args.run_keyword)
if not os.path.exists(out_dir):
    raise Exception(f"Experiment folder {out_dir} not found. Check arg "
                    f"'run_keyword'.")

# Load the model
model = torch.load(os.path.join(out_dir, 'model.save'))
# model = C_GAE()
if args.cuda:
    model.cuda()

model.eval()


def get_trg_shift_dist():
    return 1 + model.context_length // 2


file_loader = torch.utils.data.DataLoader(
    CQT(files, trg_shift=get_trg_shift_dist(), block_size=sys.maxsize,
        allow_diff_shapes=True, n_bins=args.n_bins,
        bins_per_octave=args.bins_per_oct,
        fmin=args.fmin, hop_length=args.hop_length,
        convolutional=True, refresh_cache=args.refresh_cache,
        cache_fn="trans.pyc.bz"),
    batch_size=1, shuffle=True, **kwargs)


def save_invariant():
    """
    Converts and saves

    :param epoch: the current training epoch
    """
    for batch_idx, (x, y) in enumerate(file_loader):
        filename = os.path.basename(files[batch_idx])
        filename = filename.strip('\n')

        x = cuda_variable(x)
        y = cuda_variable(y)

        x = F.dropout(x, training=False)

        # calculate mapping of untransposed data
        m = model.mapping_(x, y)

        out_fn = os.path.join(out_dir, filename + ".invar.pyc.bz")
        save_pyc_bz(np.transpose(to_numpy(m)[0, :, :, 0]), out_fn)
        print(f"Written invariant representation to {out_fn}")


save_invariant()
