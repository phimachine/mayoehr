import torch.nn as nn
import archi.param as param
import torch.nn.functional as F

class Interface(nn.Module):
    # an interface processor that takes all the interface output
    # and processes them

    def __init__(self):
        super(Interface, self).__init__()
        self.logSigmoid=nn.LogSigmoid()
        self.Sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax(dim=0)
        
    def forward(self, interface_input):
        last_index=param.W*param.R

        # Read keys, each W dimensions, [W*R] in total
        # no processing needed
        # this is the address keys, not the contents
        read_keys=interface_input[:,0:last_index].view(param.bs,param.W,param.R).clone()

        # Read strengths, [R]
        # 1 to infinity
        # slightly different equation from the paper, should be okay
        read_strengths=interface_input[:,last_index:last_index+param.R].clone()
        last_index=last_index+param.R
        read_strengths=1-self.logSigmoid(read_strengths)

        # Write key, [W]
        write_key=interface_input[:,last_index:last_index+param.W].clone()
        last_index=last_index+param.W

        # write strength beta, [1]
        write_strength=interface_input[:,last_index:last_index+1].clone()
        last_index=last_index+1
        write_strength=1-self.logSigmoid(write_strength)

        # erase strength, [W]
        erase_vector=interface_input[:,last_index:last_index+param.W].clone()
        last_index=last_index+param.W
        erase_vector=self.Sigmoid(erase_vector)

        # write vector, [W]
        write_vector=interface_input[:,last_index:last_index+param.W].clone()
        last_index=last_index+param.W

        # R free gates? [R] TODO what is this?
        free_gates=interface_input[:,last_index:last_index+param.R].clone()

        last_index=last_index+param.R
        free_gates=self.Sigmoid(free_gates)

        # allocation gate [1]
        allocation_gate=interface_input[:,last_index:last_index+1].clone()
        last_index=last_index+1
        allocation_gate=self.Sigmoid(allocation_gate)

        # write gate [1]
        write_gate=interface_input[:,last_index:last_index+1].clone()
        last_index=last_index+1
        write_gate=self.Sigmoid(write_gate)

        # read modes [R,3]
        read_modes=interface_input[:,last_index:last_index+param.R*3].clone()
        read_modes=self.softmax(read_modes)
        read_modes=read_modes.view(param.bs,param.R,3)

        # total dimension: param.W*param.R+3*param.W+5*param.R+3
        return read_keys, read_strengths, write_key, write_strength, \
               erase_vector, write_vector, free_gates, allocation_gate, \
               write_gate, read_modes
