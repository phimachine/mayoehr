# reference Methods, controller network

from archi.param import *
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import archi.param as param
import math
from torch.nn.modules.rnn import LSTM


class Controller(nn.Module):
    """
        Deep RNN model that passes on layer by layer
    """
    def __init__(self):
        super(Controller, self).__init__()
        self.RNN_list=nn.ModuleList()
        for _ in range(param.L):
            self.RNN_list.append(RNN_Unit())
        self.hidden_previous_timestep=torch.Tensor(param.L,param.h).zero_()
        self.W_y=nn.Linear(param.L*param.h,param.v_t)
        self.W_E=nn.Linear(param.L*param.h,param.E_t)

    def forward(self, input_x):
        '''

        :param input_x: raw input concatenated with flattened memory input
        :return:
        '''
        hidden_previous_layer=torch.Tensor(param.h).zero_()
        hidden_this_timestep=torch.Tensor(param.L,param.h)
        for i in range(param.L):
            hidden_output=self.RNN_list[i](input_x, self.hidden_previous_timestep[i],
                             hidden_previous_layer)
            hidden_this_timestep[i]=hidden_output
            hidden_previous_layer=hidden_output

        flat=hidden_this_timestep.view([param.L*param.h])
        output=self.W_y(flat)
        interface=self.W_E(flat)
        self.hidden_previous_timestep=hidden_this_timestep
        return output, interface

    def reset_parameters(self):
        for module in self.RNN_list:
            # this should iterate over RNN_Units only
            module.reset_parameters()
        self.W_y.reset_parameters()
        self.W_E.reset_parameters()

    def new_sequence_reset(self):
        # TODO review this function. Is there more that need to be reset with a new seq?
        self.hidden_previous_timestep.zero_()

class RNN_Unit(nn.Module):
    """
    A single unit of deep RNN
    """

    def __init__(self):
        super(RNN_Unit, self).__init__()
        self.W_input=nn.Linear(param.x+2*param.h,param.h)
        self.b_input=Parameter(torch.Tensor(param.h))
        self.W_forget=nn.Linear(param.x+2*param.h,param.h)
        self.b_forget=Parameter(torch.Tensor(param.h))
        self.W_output=nn.Linear(param.x+2*param.h,param.h)
        self.b_output=nn.Parameter(torch.Tensor(param.h))
        self.W_state=nn.Linear(param.x+2*param.h,param.h)
        self.b_state=nn.Parameter(torch.Tensor(param.h))

        # state or cell, initialized in place to zero.
        self.old_state=Parameter(torch.Tensor(param.h).zero_())

        # critical
        self.reset_parameters()

    def reset_parameters(self):
        # initialized the way pytorch LSTM is initialized, from normal
        # initial state and cell are empty

        # if this is not run, any output might be nan

        stdv= 1.0 /math.sqrt(param.h)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
        for module in self.children():
            # do not use self.modules(), because it would be recursive
            module.reset_parameters()


    def forward(self,input_x,previous_time,previous_layer):
        # a hidden unit outputs a hidden output new_hidden.
        # state also changes, but it's hidden inside a hidden unit.

        semicolon_input=torch.cat([input_x,previous_time,previous_layer])

        # 5 equations
        input_gate=torch.sigmoid(self.W_input(semicolon_input)+self.b_input)
        forget_gate=torch.sigmoid(self.W_forget(semicolon_input)+self.b_forget)
        new_state=forget_gate * self.old_state + input_gate * \
                   torch.tanh(self.W_state(semicolon_input)+ self.b_state)
        output_gate=torch.sigmoid(self.W_output(semicolon_input)+self.b_output)
        new_hidden=output_gate*torch.tanh(new_state)

        return new_hidden



