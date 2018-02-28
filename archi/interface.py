import torch.nn as nn
from archi.parameters import *

class interface(nn.Module):
    # an interface processor that takes all the interface output
    # and processes them


    def __init__(self):
        super(interface, self).__init__()

    def forward(self, interface_input):
        # TODO no initiation on assigned tensor here, see if it works

        last_index=param_W*param_R

        # Read keys, each W dimensions, [W*R] in total
        # no processing needed
        # this is the address keys, not the contents
        read_keys=interface_input[0:last_index].view(param_W,-1)

        # Read strengths, [R]
        # 1 to infinity
        # slightly different equation from the paper, should be okay
        read_strengths=interface_input[last_index:last_index+param_R]
        last_index=last_index+param_R
        read_strengths=1-nn.LogSigmoid(read_strengths)

        # Write key, [W]
        write_key=interface_input[last_index:last_index+param_W]
        last_index=last_index+param_W

        # write strength beta, [1]
        write_strength=interface_input[last_index:last_index+1]
        last_index=last_index+1
        write_strength=1-nn.LogSigmoid(write_strength)

        # erase strength, [W]
        erase_vector=interface_input[last_index:last_index+param_W]
        last_index=last_index+param_W
        erase_vector=nn.Sigmoid(erase_vector)

        # write vector, [W]
        write_vector=interface[last_index:last_index+param_W]
        last_index=last_index+param_W

        # R free gates? [R] TODO what is this?
        free_gates=interface[last_index:last_index+param_R]
        last_index=last_index+param_R
        free_gates=nn.Sigmoid(free_gates)

        # allocation gate [1]
        allocation_gate=interface[last_index:last_index+1]
        last_index=last_index+1
        allocation_gate=nn.Sigmoid(allocation_gate)

        # write gate [1]
        write_gate=interface[last_index:last_index+1]
        last_index=last_index+1
        write_gate=nn.Sigmoid(write_gate)

        # read modes [R]
        read_modes=interface[last_index:last_index+param_R]
        read_modes=nn.Softmax(read_modes)

        # total dimension: param_W*param_R+3*param_W+5*param_R+3
        # TODO I count a param_W feweer than it's supposed to have.
        return read_keys, read_strengths, write_key, write_strength, \
               erase_vector, write_vector, free_gates, allocation_gate, \
               write_gate, read_modes
