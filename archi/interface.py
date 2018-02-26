import torch.nn as nn
from archi.parameters import *

class interface(nn.Module):
    # an interface processor that takes all the interface output
    # and processes them


    def __init__(self):
        super(interface, self).__init__()

    def forward(self, interface_input):
        # TODO no initiation here, see if it works

        last_index=param_W*param_R-1

        # R read keys, each W dimensions, W*R in total
        # no processing needed
        read_keys=interface_input[0:last_index].view(param_W,-1)

        # R read strengths
        # 1 to infinity
        # slightly different equation from the paper, should be okay
        read_strengths=interface_input[last_index+1:param_W*param_R+param_R]
        read_strengths=1-nn.LogSigmoid(read_strengths)

        # one write key, W
        write_key=interface_input[param]

        return read_keys, read_strengths,