"""
Created on April 13, 2018

@author: Gaetan Hadjeres & Stefan Lattner

Sony CSL Paris, France
"""

import torch
from torch.autograd import Variable

def cuda_tensor(data):
    if torch.cuda.is_available():
        return torch.FloatTensor(data).cuda()
    else:
        return torch.FloatTensor(data)

def cuda_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        return Variable(tensor.cuda(), volatile=volatile)
    else:
        return Variable(tensor, volatile=volatile)


def to_numpy(variable):
    try:
        if torch.cuda.is_available():
            return variable.data.cpu().numpy()
        else:
            return variable.data.numpy()
    except:
        return variable.numpy()

