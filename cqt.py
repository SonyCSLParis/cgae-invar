"""
Created on April 13, 2018

@author: Stefan Lattner & Maarten Grachten

Sony CSL Paris, France
Institute for Computational Perception, Johannes Kepler University, Linz
Austrian Research Institute for Artificial Intelligence, Vienna

"""

import bz2
import logging
import os
import pickle

import librosa
import numpy as np
import torch
import torch.utils.data as data

from pickle import UnpicklingError

LOGGER = logging.getLogger(__name__)


def save_pyc_bz(data, fn):
    """
    Saves data to file (bz2 compressed)

    :param data: data to save
    :param fn: file name of dumped data
    """
    pickle.dump(data, bz2.BZ2File(fn, 'w'))


def load_pyc_bz(fn):
    """
    Loads data from file (bz2 compressed)

    :param fn: file name of dumped data
    :return: loaded data
    """
    return pickle.load(bz2.BZ2File(fn, 'r'))


def cached(cache_fn, func, args=(), kwargs={}, refresh_cache=False):
    """
    If `cache_fn` exists, return the unpickled contents of that file
    (the cache file is treated as a bzipped pickle file). If this
    fails, compute `func`(*`args`), pickle the result to `cache_fn`,
    and return the result.

    Parameters
    ----------

    func : function
        function to compute

    args : tuple
        argument for which to evaluate `func`

    cache_fn : str
        file name to load the computed value `func`(*`args`) from

    refresh_cache : boolean
        if True, ignore the cache file, compute function, and store the result
        in the cache file

    Returns
    -------

    object

        the result of `func`(*`args`)

    """
    result = None
    if cache_fn is not None and os.path.exists(cache_fn):
        if refresh_cache:
            os.remove(cache_fn)
        else:
            try:
                LOGGER.info(f"Loading cache file {cache_fn}...")
                result = load_pyc_bz(cache_fn)
            except UnpicklingError as e:
                LOGGER.error(('The file {0} exists, but cannot be unpickled.'
                              'Is it readable? Is this a pickle file? Try '
                              'with numpy..'
                              '').format(cache_fn))
                try:
                    result = np.load(cache_fn)
                except Exception as g:
                    LOGGER.error("Did not work, either.")
                    raise e

    if result is None:
        result = func(*args, **kwargs)
        if cache_fn is not None:
            try:
                save_pyc_bz(result, cache_fn)
            except Exception as e:
                LOGGER.error("Could not save, try with numpy..")
                try:
                    np.save(cache_fn, result)
                except Exception as g:
                    LOGGER.error("Did not work, either.")
                    raise e
    return result


def contrast_normalize(x, axis=1):
    """
    Performs contrast normalization (zero mean, unit variance)
    along the given axis.

    :param x: array to normalize
    :param axis: normalize along that axis
    :return: contrast-normalized array
    """
    means = np.mean(x, axis=axis, keepdims=True)
    x -= means
    stds = np.sqrt(np.sum(x ** 2, axis=axis, keepdims=True))
    stds[stds < 1e-8] = 1
    x = x / stds
    return x


class CQT(data.Dataset):

    def __init__(self, filelist, trg_shift=1, block_size=1024,
                 n_bins=120, bins_per_octave=24, fmin=65.4, hop_length=448,
                 convolutional=True, refresh_cache=False,
                 cache_fn="cqt_cache.pyc.bz",
                 allow_diff_shapes=False, padded=False):
        """
        Constructor for Constant-Q-Transform dataset

        :param filelist:        list of audio file names (str)
        :param trg_shift:       target == input shifted by [-trg_shift] steps,
                                blocks are shortened accordingly
        :param block_size:      length of one instance in a batch
        :param convolutional:   instance dim=1 or instance dim=3
        :param refresh_cache:   when True recalculate and save to cache file
                                when False loads from cache file when available
        """
        self.trg_shift = trg_shift
        self.block_size = block_size
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin
        self.hop_length = hop_length
        self.convolutional = convolutional
        self.allow_diff_shapes = allow_diff_shapes
        self.padded = padded
        self.data_cqt, self.lengths = cached(cache_fn,
                                             self.files_to_cqt, (filelist,),
                                             refresh_cache=refresh_cache)

    def __getitem__(self, index):
        if self.convolutional:
            x = torch.FloatTensor(self.data_cqt[index][0][None, :, :])
            y = torch.FloatTensor(self.data_cqt[index][1][None, :, :])
        else:
            x = torch.FloatTensor(self.data_cqt[index][0])
            y = torch.FloatTensor(self.data_cqt[index][1])

        return x, y

    def __len__(self):
        return len(self.data_cqt)

    def files_to_cqt(self, filelist):
        """
        Transforms audio files into blocks of constant-Q-transformed
        spectrograms, in two versions mutually shifted in time to be
        used for input and target in prediction setting.

        :param filelist: list of filenames (str)
        :return: CQT representation in equally sized blocks
        """
        data_cqt = []
        lengths = []
        max_length = 0
        for file in filelist:
            file = file.strip('\n')
            print(f"loading file {file}")
            audio = librosa.load(file)
            print(f"transforming {file} to cqt")
            cqt = librosa.cqt(audio[0], n_bins=self.n_bins,
                              bins_per_octave=self.bins_per_octave,
                              fmin=self.fmin, hop_length=self.hop_length)
            cqt = np.transpose(librosa.magphase(cqt)[0])
            cqt = contrast_normalize(cqt)

            if self.trg_shift is not 0:
                # pad to create input and target
                cqt_x = np.concatenate((cqt[:self.trg_shift], cqt))
                cqt_y = np.concatenate((cqt, cqt[-self.trg_shift:]))
            else:
                cqt_x = cqt_y = cqt

            if len(cqt) > max_length:
                max_length = len(cqt)

            for i in range(0, len(cqt), self.block_size):
                if not self.allow_diff_shapes and i + self.block_size > \
                        len(cqt):
                    break

                cqt_x_block = cqt_x[i:i + self.block_size]
                cqt_y_block = cqt_y[i:i + self.block_size]

                length = len(cqt_x_block)
                lengths.append(length)

                if self.padded:
                    diff = self.block_size - length
                    if diff > 0:
                        pad = np.zeros((diff, *cqt_x_block.shape[1:]))
                        cqt_x_block = np.concatenate((cqt_x_block, pad))
                        cqt_y_block = np.concatenate((cqt_y_block, pad))

                if self.convolutional:
                    data_cqt.append([cqt_x_block, cqt_y_block])
                else:
                    data_cqt.append([
                        np.reshape(cqt_x_block, -1),
                        np.reshape(cqt_y_block, -1)]
                    )

        if len(data_cqt) == 0:
            LOGGER.warning("No data added to CQT Dataset! Is blocksize too high?")

        if max_length < self.block_size and self.padded:
            LOGGER.warning(f"Consider reducing block size of {self.block_size} to {max_length}!")

        return data_cqt, lengths
