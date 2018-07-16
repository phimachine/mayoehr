# reference Methods, controller network

from archi.param import *
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import archi.param as param
import math
from torch.nn.modules.rnn import LSTM
from torch.autograd import Variable


class Controller(nn.LSTM):

    def __init__(self):
        super(Controller, self).__init__(input_size=param.x + param.R * param.W,
                                         hidden_size=param.h,
                                         num_layers=param.L,
                                         bias=True,
                                         batch_first=True,
                                         dropout=True,
                                         bidirectional=False)
        # self.last_hidden = Variable(torch.Tensor(param.L, param.bs, param.h).zero_().cuda())
        self.last_hidden = Variable(torch.Tensor(param.bs, param.L, param.h).zero_().cuda())
        self.last_cell = Variable(torch.Tensor(param.L, param.bs, param.h).zero_().cuda())
        self.W_y = Variable(torch.Tensor(param.L * param.h, param.v_t)).cuda()
        self.W_E = Variable(torch.Tensor(param.L * param.h, param.E_t)).cuda()
        self.b_y = Variable(torch.Tensor(param.v_t)).cuda()
        self.b_E = Variable(torch.Tensor(param.E_t)).cuda()

    def forward(self, input):
        # expanded to fake a one-sequence
        output, (hidden, cell) = super(Controller, self).forward(input, (self.last_hidden, self.last_cell))

        self.last_hidden = hidden
        self.last_cell = cell

        flat_hidden = hidden.view((param.bs, param.L * param.h))
        output = torch.matmul(flat_hidden, self.W_y) + self.b_y
        output=torch.nn
        interface = torch.matmul(flat_hidden, self.W_E) + self.b_E

        return output, interface

    def new_sequence_reset(self):
        self.last_hidden = Variable(torch.Tensor(2, param.L, param.bs, param.h).zero_().cuda())
        self.last_cell = Variable(self.last_cell.data)
        self.W_y = Variable(self.W_y.data)
        self.b_y = Variable(self.b_y.data)
        self.W_E = Variable(self.W_E.data)
        self.b_E = Variable(self.b_E.data)

#
#
#
# class Controller(nn.Module):
#     """
#         Deep RNN model that passes on layer by layer
#     """
#     def __init__(self):
#         super(Controller, self).__init__()
#         self.RNN_list=nn.ModuleList()
#         for _ in range(param.L):
#             self.RNN_list.append(RNN_Unit())
#         self.hidden_previous_timestep=Variable(torch.Tensor(param.bs,param.L,param.h).zero_().cuda())
#         self.W_y=nn.Linear(param.L*param.h,param.v_t)
#         self.W_E=nn.Linear(param.L*param.h,param.E_t)
#
#     def forward(self, input_x):
#         '''
#
#         :param input_x: raw input concatenated with flattened memory input
#         :return:
#         '''
#         hidden_previous_layer=Variable(torch.Tensor(param.bs,param.h).zero_().cuda())
#         hidden_this_timestep=Variable(torch.Tensor(param.bs,param.L,param.h).cuda())
#         for i in range(param.L):
#             hidden_output=self.RNN_list[i](input_x, self.hidden_previous_timestep[:,i,:],
#                                            hidden_previous_layer)
#             hidden_this_timestep[:,i,:]=hidden_output
#             hidden_previous_layer=hidden_output
#
#         flat_hidden=hidden_this_timestep.view((param.bs,param.L*param.h))
#         output=self.W_y(flat_hidden)
#         interface=self.W_E(flat_hidden)
#         self.hidden_previous_timestep=hidden_this_timestep
#         return output, interface
#
#     def reset_parameters(self):
#         for module in self.RNN_list:
#             # this should iterate over RNN_Units only
#             module.reset_parameters()
#         self.W_y.reset_parameters()
#         self.W_E.reset_parameters()
#
#     def new_sequence_reset(self):
#         self.hidden_previous_timestep=Variable(torch.Tensor(param.bs,param.L,param.h).zero_().cuda())
#         for RNN in self.RNN_list:
#             RNN.new_sequence_reset()
#         self.hidden_previous_timestep.detach()
#         self.W_y.weight.detach()
#         self.W_y.bias.detach()
#         self.W_E.weight.detach()
#         self.W_E.bias.detach()
#
# class RNN_Unit(nn.Module):
#     """
#     LSTM
#     """
#
#     def __init__(self):
#         super(RNN_Unit, self).__init__()
#         self.W_input=nn.Linear(param.x+param.R*param.W+2*param.h,param.h)
#         self.W_forget=nn.Linear(param.x+param.R*param.W+2*param.h,param.h)
#         self.W_output=nn.Linear(param.x+param.R*param.W+2*param.h,param.h)
#         self.W_state=nn.Linear(param.x+param.R*param.W+2*param.h,param.h)
#
#         self.old_state=Variable(torch.Tensor(param.bs,param.h).zero_().cuda())
#
#
#     def reset_parameters(self):
#         # initialized the way pytorch LSTM is initialized, from normal
#         # initial state and cell are empty
#
#         # if this is not run, any output might be nan
#         stdv= 1.0 /math.sqrt(param.h)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv,stdv)
#         for module in self.children():
#             # do not use self.modules(), because it would be recursive
#             module.reset_parameters()
#
#
#     def forward(self,input_x,previous_time,previous_layer):
#         # a hidden unit outputs a hidden output new_hidden.
#         # state also changes, but it's hidden inside a hidden unit.
#
#         # I think .data call is safe whenever new Variable should be initated.
#         # Unless I wish to have the gradients recursively flow back to the beginning of history
#         # I do not wish so.
#         semicolon_input=torch.cat((input_x,previous_time,previous_layer),dim=1)
#
#         # 5 equations
#         input_gate=torch.sigmoid(self.W_input(semicolon_input))
#         forget_gate=torch.sigmoid(self.W_forget(semicolon_input))
#         new_state=forget_gate * self.old_state + input_gate * \
#                   torch.tanh(self.W_state(semicolon_input))
#         output_gate=torch.sigmoid(self.W_output(semicolon_input))
#         new_hidden=output_gate*torch.tanh(new_state)
#
#         # TODO Warning: needs to assign, not sure if this is right
#         # Good warning, I have changed the assignment and I hope this now works better.
#         self.old_state=new_state
#
#         return new_hidden
#
#     def new_sequence_reset(self):
#         self.old_state=Variable(torch.Tensor(param.bs,param.h).zero_().cuda())
#         self.W_input.weight.detach()
#         self.W_input.bias.detach()
#         self.W_forget.weight.detach()
#         self.W_forget.bias.detach()
#         self.W_output.weight.detach()
#         self.W_output.bias.detach()
#         self.W_state.weight.detach()
#         self.W_state.bias.detach()
#         self.old_state.detach()
#         print('reset controller LSTM')
