import torch
from torch import nn
from archi.interface import Interface
from archi.controller import Controller
from archi.memory import Memory
import archi.param as param
import pdb

class Computer(nn.Module):

    def __init__(self):
        super(Computer, self).__init__()
        self.memory=Memory()
        self.controller=Controller()
        self.interface=Interface()
        self.last_read_vector=torch.Tensor(param.bs,param.W, param.R)

    def forward(self, input):
        input_x_t=torch.cat((input,self.last_read_vector.view(param.bs,-1)),dim=1)
        output, interface=self.controller(input_x_t)
        if torch.isnan(output).any():
            pdb.set_trace()
            raise ValueError("nan is found in the batch output.")
        interface_output_tuple=self.interface(interface)
        self.last_read_vector=self.memory(*interface_output_tuple)
        return output

    def reset_parameters(self):
        self.memory.reset_parameters()
        self.controller.reset_parameters()
        self.last_read_vector.zero_()
        # no parameter in interface

    def new_sequence_reset(self):
        # I have not found a reference to this function, but I think it's reasonable
        # to reset the values that depends on a particular sequence.
        self.controller.new_sequence_reset()
        self.memory.new_sequence_reset()
        self.last_read_vector.zero_()