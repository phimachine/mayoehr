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
        for _ in param.L:
            self.RNN_list.append(RNN_Unit())
        self.hidden_previous_timestep=torch.Tensor(param.L*param.h).zero_()
        self.W_y=nn.Linear(param.L*param.h,param.v_t)
        self.W_E=nn.Linear(param.L*param.h,param.E_t)

    def forward(self, input_x):
        # the units cannot be calculated with a big matrix.
        # since each unit will need previous unit's output, the calculation must
        # be time-differentiated
        # not sure whether this is the most efficient loop implementation.
        hidden_previous_layer=torch.Tensor(param.h).zero_()
        hidden_this_timestep=torch.Tensor(param.L*param.h)
        for i in param.L:
            hidden_output=self.RNN_list[i](input_x, self.all_previous_time[i],
                             hidden_previous_layer)
            # TODO verify this line, see if slicing is possible. torch.narrow()
            hidden_this_timestep[i*param.h:(i+1)*param.h]=hidden_output
            hidden_previous_layer=hidden_output

        output=self.W.y(hidden_this_timestep)
        interface=self.W_E(hidden_this_timestep)
        self.hidden_previous_timestep=hidden_this_timestep
        return output, interface


class RNN_Unit(nn.Module):
    """
    A single unit of deep RNN
    """

    def __init__(self):
        super(RNN_Unit, self).__init__()
        self.W_input=nn.Linear(param.s,param.x+2*param.h)
        self.b_input=Parameter(torch.Tensor(param.s))
        self.W_forget=nn.Linear(param.s,param.x+2*param.h)
        self.b_forget=Parameter(torch.Tensor(param.s))
        self.W_output=nn.Linear(param.s,param.x+2*param.h)
        self.b_output=nn.Parameter(torch.Tensor(param.s))
        self.W_state=nn.Linear(param.s,param.x+2*param.h)
        self.b_state=nn.Parameter(torch.Tensor(param.s))

        # state or cell, initialized in place to zero.
        self.old_state=Parameter(torch.Tensor(param.s).zero_())

    def reset_parameters(self):
        # initialized the way pytorch LSTM is initialized, from normal
        # initial state and cell are empty
        stdv= 1.0 /math.sqrt(param.h)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)


    def forward(self,input_x,previous_time,previous_layer):
        # a hidden unit outputs a hidden output new_hidden.
        # state also changes, but it's hidden inside a hidden unit.

        semicolon_input=torch.cat(input_x,previous_time,previous_layer)

        # 5 equations
        input_gate=torch.nn.Sigmoid(self.W_input(semicolon_input)+self.b_input)
        forget_gate=torch.nn.Sigmoid(self.W_forget(semicolon_input)+self.b_forget)
        new_state=forget_gate * self.old_state + input_gate * \
                   nn.Tanh(self.W_state(semicolon_input)+ self.b_state)
        output_gate=torch.nn.Sigmoid(self.W_output(semicolon_input)+self.b_output)
        new_hidden=output_gate*nn.Tanh(new_state)

        return new_hidden



