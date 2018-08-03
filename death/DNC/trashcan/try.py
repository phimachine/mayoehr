from archi.controller import RNN_Unit
from archi.computer import Computer
from archi.param import *
from archi.interface import Interface
from archi.controller import MyController
from archi.memory import Memory
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import archi.param as param
import math
from torch.nn.modules.rnn import LSTM
from torch.autograd import Variable
#
# hello=RNN_Unit()
# print(list(hello.parameters()))
#
# class hello2(nn.Module):
#     def __init__(self):
#         super(hello2, self).__init__()
#         self.haha=Parameter(torch.Tensor([1,2,3])).cuda()
#         self.haha.data.uniform_(-1,1)
#
#
# maybe=hello2()
# print(list(maybe.parameters()))

# apparently you cannot call cuda at parameter.

hello=Computer()
print(list(hello.children()))
print(list(hello.parameters()))

hello=MyController()
hello.
