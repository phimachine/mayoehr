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
        interface_output_tuple=self.interface(interface)
        self.last_read_vector=self.memory(*interface_output_tuple)
        if torch.isnan(self.last_read_vector).any():
            read_keys, read_strengths, write_key, write_strength, \
            erase_vector, write_vector, free_gates, allocation_gate, \
            write_gate, read_modes = interface_output_tuple
            allocation_weighting = self.memory.allocation_weighting()
            write_weighting = self.memory.write_weighting(write_key, write_strength,
                                                   allocation_gate, write_gate, allocation_weighting)
            self.memory.write_to_memory(write_weighting, erase_vector, write_vector)
            # update some
            memory_retention = self.memory.memory_retention(free_gates)
            self.memory.update_usage_vector(write_weighting, memory_retention)
            self.memory.update_temporal_linkage_matrix(write_weighting)
            self.memory.update_precedence_weighting(write_weighting)

            forward_weighting = self.memory.forward_weighting()
            backward_weighting = self.memory.backward_weighting()

            read_weightings = self.memory.read_weightings(forward_weighting, backward_weighting, read_keys, read_strengths,
                                                   read_modes)
            # read from memory last, a new modification.
            read_vectors = self.memory.read_memory(read_weightings)
            raise ValueError("nan is found.")
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