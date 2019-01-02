"""
We will modify four things:
Memory data structure.
The read operation.
The write operation.
The memory linkage computation.
This script should base on Frankenstein2
"""

import torch
from torch import nn
import pdb
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity, softmax, normalize
from torch.nn.parameter import Parameter
import math
import numpy as np
import traceback
import os
from os.path import abspath
from pathlib import Path
import pickle

debug = True

def sv(var):
    return var.data.cpu().numpy()

def test_simplex_bound(tensor, dim=1):
    # it's impossible to deal with dimensions
    # we will default to test dim 1 of 2-dim (x, y),
    # so that for every x, y is simplex bound

    if dim != 1:
        raise DeprecationWarning("no longer accepts dim other othan one")
        raise NotImplementedError
    t = tensor.contiguous()
    if (t.sum(1) - 1 > 1e-6).any() or (t.sum(1) < -1e-6).any() or (t < 0).any() or (t > 1).any():
        raise ValueError("test simplex bound failed")
    if (t != t).any():
        raise ValueError('test simple bound failed due to NA')
    return True


class SAM(nn.Module):

    def __init__(self):
        super(SAM, self).__init__()

