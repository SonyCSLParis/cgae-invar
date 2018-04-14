"""
Created on April 13, 2018

@author: Stefan Lattner

Sony CSL Paris, France
Institute for Computational Perception, Johannes Kepler University, Linz
Austrian Research Institute for Artificial Intelligence, Vienna

"""

import torch


def lee_loss(act):
    """
    Lee sparsity and selectivity regularization on hidden activations
    :param act: hidden unit activations
    :return: loss
    """
    sparsity = torch.sum((act ** 2).mean(3).mean(2).mean(0)).sum()
    selectivity = torch.sum((act ** 2).mean(3).mean(1).mean(0)).sum()
    return sparsity + selectivity


def l2_loss(weight):
    """
    L2 weight regularization
    :param weight: weight to regularize
    :return: loss
    """
    return torch.sum(weight ** 2)