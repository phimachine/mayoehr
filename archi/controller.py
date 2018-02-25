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


    def forward(self, input):
        """

        :param input:
        x_t: [X], input vector at time step t

        

        :return:output:
        v_t

        """


        output=None
        interface=None
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

        self.old_state=Parameter(torch.Tensor(param_s))

    def reset_parameters(self):
        pass

    def forward(self,input_x,previous_time,previous_layer):
        semicolon_input=torch.cat(input_x,previous_time,previous_layer)
        input_gate=torch.nn.Sigmoid(self.W_input(semicolon_input)+self.b_input)
        forget_gate=torch.nn.Sigmoid(self.W_forget(semicolon_input)+self.b_forget)
        output_gate=torch.nn.Sigmoid(self.W_output(semicolon_input)+self.b_output)

        new_state=forget_gate * self.old_state + input_gate * \
                   nn.Tanh(self.W_state(semicolon_input)+ self.b_state)
        new_hidden=output_gate*nn.Tanh(new_state)




