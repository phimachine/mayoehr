import torch.nn as nn
import archi.param as param

class interface(nn.Module):
    # an interface processor that takes all the interface output
    # and processes them


    def __init__(self):
        super(interface, self).__init__()

    def forward(self, interface_input):
        # TODO no initiation on assigned tensor here, see if it works

        last_index=param.W*param.R

        # Read keys, each W dimensions, [W*R] in total
        # no processing needed
        # this is the address keys, not the contents
        read_keys=interface_input[0:last_index].view(param.W,-1)

        # Read strengths, [R]
        # 1 to infinity
        # slightly different equation from the paper, should be okay
        read_strengths=interface_input[last_index:last_index+param.R]
        last_index=last_index+param.R
        read_strengths=1-nn.LogSigmoid(read_strengths)

        # Write key, [W]
        write_key=interface_input[last_index:last_index+param.W]
        last_index=last_index+param.W

        # write strength beta, [1]
        write_strength=interface_input[last_index:last_index+1]
        last_index=last_index+1
        write_strength=1-nn.LogSigmoid(write_strength)

        # erase strength, [W]
        erase_vector=interface_input[last_index:last_index+param.W]
        last_index=last_index+param.W
        erase_vector=nn.Sigmoid(erase_vector)

        # write vector, [W]
        write_vector=interface[last_index:last_index+param.W]
        last_index=last_index+param.W

        # R free gates? [R]
        free_gates=interface[last_index:last_index+param.R]
        last_index=last_index+param.R
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
        read_modes=interface[last_index:last_index+param.R]
        read_modes=nn.Softmax(read_modes)

        # total dimension: param.W*param.R+3*param.W+5*param.R+3
        # TODO I count a param.W fewer than it's supposed to have.
        return read_keys, read_strengths, write_key, write_strength, \
               erase_vector, write_vector, free_gates, allocation_gate, \
               write_gate, read_modes


# Maybe not, this means all the values will be stored in objects. No.
# class Detupler(interface):
#
#     def __init__(self,interface):
#         self.read_keys, self.read_strengths, self.write_key, self.write_strength, \
#         self.erase_vector, self.write_vector, self.free_gates, self.allocation_gate, \
#         self.write_gate, self.read_modes = interface
#
#

