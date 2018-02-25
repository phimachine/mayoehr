# reference Methods, controller network

from archi.parameters import *
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Controller(nn.Module):
    """
        Deep RNN model that passes on layer by layer

    """
    def __init__(self):
        super(Controller, self).__init__()
        self.RNN_list=nn.ModuleList()
        for _ in param_L:
            self.RNN_list.append(RNN_Unit())
        self.hidden_previous_timestep=torch.Tensor(param_L*param_h).zero_()
        self.W_y=nn.Linear(param_L*param_h,param_v_t)
        self.W_E=nn.Linear(param_L*param_h,param_E_t)

    def forward(self, input_x):
        # the units cannot be calculated with a big matrix.
        # since each unit will need previous unit's output, the calculation must
        # be time-differentiated
        # not sure whether this is the most efficient loop implementation.
        hidden_previous_layer=torch.Tensor(param_h).zero_()
        hidden_this_timestep=torch.Tensor(param_L*param_h)
        for i in param_L:
            hidden_output=self.RNN_list[i](input_x, self.all_previous_time[i],
                             hidden_previous_layer)
            # TODO verify this line, see if slicing is possible. torch.narrow()
            hidden_this_timestep[i*param_h:(i+1)*param_h]=hidden_output
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
        self.W_input=nn.Linear(param_s,param_x+2*param_h)
        self.b_input=Parameter(torch.Tensor(param_s))
        self.W_forget=nn.Linear(param_s,param_x+2*param_h)
        self.b_forget=Parameter(torch.Tensor(param_s))
        self.W_output=nn.Linear(param_s,param_x+2*param_h)
        self.b_output=nn.Parameter(torch.tensor(param_s))
        self.W_state=nn.Linear(param_s,param_x+2*param_h)
        self.b_state=nn.Parameter(torch.tensor(param_s))

        # state or cell, initialized in place to zero.
        self.old_state=Parameter(torch.Tensor(param_s).zero_())

    def reset_parameters(self):
        pass

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



