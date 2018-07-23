# reference Methods, controller network

from archi.param import *
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import archi.param as param
import math
from torch.nn.modules.rnn import LSTM
from torch.autograd import Variable


# initiation and requires grad examined.
#
# class Controller(nn.LSTM):
#
#     def __init__(self):
#         super(Controller, self).__init__(input_size=param.x + param.R * param.W,
#                                          hidden_size=param.h,
#                                          num_layers=param.L,
#                                          bias=True,
#                                          batch_first=True,
#                                          dropout=True,
#                                          bidirectional=False)
#         self.reset_parameters()
#         self.last_hidden = Variable(torch.Tensor(param.L, param.bs, param.h).zero_())
#         # self.last_hidden = Variable(torch.Tensor(param.bs, param.L, param.h).zero_())
#         self.last_cell = Variable(torch.Tensor(param.L, param.bs, param.h).zero_())
#         self.W_y = Parameter(torch.Tensor(param.L * param.h, param.v_t),requires_grad=True)
#         self.W_E = Parameter(torch.Tensor(param.L * param.h, param.E_t),requires_grad=True)
#         self.b_y = Parameter(torch.Tensor(param.v_t),requires_grad=True)
#         self.b_E = Parameter(torch.Tensor(param.E_t),requires_grad=True)
#         wls=[self.W_y,self.W_E,self.b_y,self.b_E]
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for w in wls:
#             w.data.uniform_(-stdv, stdv)
#
#     def forward(self, input):
#         # expanded to fake a one-sequence
#         output, (hidden, cell) = super(Controller, self).forward(input, (self.last_hidden, self.last_cell))
#
#         output=torch.nn.functional.sigmoid(output)
#
#         self.last_hidden = hidden
#         self.last_cell = cell
#
#         flat_hidden = hidden.view((param.bs, param.L * param.h))
#         output = torch.matmul(flat_hidden, self.W_y) + self.b_y
#         interface = torch.matmul(flat_hidden, self.W_E) + self.b_E
#
#         return output, interface
#
#     def new_sequence_reset(self):
#         self.last_hidden = Variable(torch.Tensor(param.L, param.bs, param.h).zero_())
#         # stateless, per paper
#         self.last_cell = Variable(torch.Tensor(param.L, param.bs, param.h).zero_())
#         # self.last_cell = Variable(self.last_cell.data)
#         self.W_y = Parameter(self.W_y.data,requires_grad=True)
#         self.b_y = Parameter(self.b_y.data,requires_grad=True)
#         self.W_E = Parameter(self.W_E.data,requires_grad=True)
#         self.b_E = Parameter(self.b_E.data,requires_grad=True)




class MyController(nn.Module):
    """
        Deep RNN model that passes on layer by layer
    """
    def __init__(self):
        super(MyController, self).__init__()
        self.RNN_list=nn.ModuleList()
        for _ in range(param.L):
            self.RNN_list.append(RNN_Unit())
        self.hidden_previous_timestep=Variable(torch.Tensor(param.bs,param.L,param.h).zero_())
        # self.W_y=nn.Linear(param.L*param.h,param.v_t)
        # self.W_E=nn.Linear(param.L*param.h,param.E_t)
        self.W_y = Parameter(torch.Tensor(param.L * param.h, param.v_t),requires_grad=True)
        self.W_E = Parameter(torch.Tensor(param.L * param.h, param.E_t),requires_grad=True)
        self.b_y = Parameter(torch.Tensor(param.v_t),requires_grad=True)
        self.b_E = Parameter(torch.Tensor(param.E_t),requires_grad=True)

        wls=(self.W_y,self.W_E,self.b_y,self.b_E)
        stdv = 1.0 / math.sqrt(param.h)
        for w in wls:
            w.data.uniform_(-stdv, stdv)

    def forward(self, input_x):
        '''

        :param input_x: raw input concatenated with flattened memory input
        :return:
        '''
        hidden_previous_layer=Variable(torch.Tensor(param.bs,param.h).zero_())
        hidden_this_timestep=Variable(torch.Tensor(param.bs,param.L,param.h))
        for i in range(param.L):
            hidden_output=self.RNN_list[i](input_x, self.hidden_previous_timestep[:,i,:],
                                           hidden_previous_layer)
            hidden_this_timestep[:,i,:]=hidden_output
            hidden_previous_layer=hidden_output

        flat_hidden=hidden_this_timestep.view((param.bs,param.L*param.h))
        output=torch.matmul(flat_hidden,self.W_y)
        interface=torch.matmul(flat_hidden,self.W_E)
        self.hidden_previous_timestep=hidden_this_timestep
        return output, interface

    def reset_parameters(self):
        for module in self.RNN_list:
            # this should iterate over RNN_Units only
            module.reset_parameters()
        self.W_y.reset_parameters()
        self.W_E.reset_parameters()

    def new_sequence_reset(self):
        self.hidden_previous_timestep=Variable(torch.Tensor(param.bs,param.L,param.h).zero_())
        for RNN in self.RNN_list:
            RNN.new_sequence_reset()
        # self.hidden_previous_timestep.detach()
        # self.W_y.weight.detach()
        # self.W_y.bias.detach()
        # self.W_E.weight.detach()
        # self.W_E.bias.detach()
        self.W_y = Parameter(self.W_y.data,requires_grad=True)
        self.b_y = Parameter(self.b_y.data,requires_grad=True)
        self.W_E = Parameter(self.W_E.data,requires_grad=True)
        self.b_E = Parameter(self.b_E.data,requires_grad=True)

class RNN_Unit(nn.Module):
    """
    LSTM
    """

    def __init__(self):
        super(RNN_Unit, self).__init__()
        # self.W_input=nn.Linear(param.x+param.R*param.W+2*param.h,param.h)
        # self.W_forget=nn.Linear(param.x+param.R*param.W+2*param.h,param.h)
        # self.W_output=nn.Linear(param.x+param.R*param.W+2*param.h,param.h)
        # self.W_state=nn.Linear(param.x+param.R*param.W+2*param.h,param.h)

        self.W_input=Parameter(torch.Tensor(param.x+param.R*param.W+2*param.h,param.h),requires_grad=True)
        self.W_forget=Parameter(torch.Tensor(param.x+param.R*param.W+2*param.h,param.h),requires_grad=True)
        self.W_output=Parameter(torch.Tensor(param.x+param.R*param.W+2*param.h,param.h),requires_grad=True)
        self.W_state=Parameter(torch.Tensor(param.x+param.R*param.W+2*param.h,param.h),requires_grad=True)

        self.b_input=Parameter(torch.Tensor(param.h),requires_grad=True)
        self.b_forget=Parameter(torch.Tensor(param.h),requires_grad=True)
        self.b_output=Parameter(torch.Tensor(param.h),requires_grad=True)
        self.b_state=Parameter(torch.Tensor(param.h),requires_grad=True)

        wls=(self.W_forget,self.W_output,self.W_input,self.W_state,self.b_input,self.b_forget,self.b_output,self.b_state)
        stdv = 1.0 / math.sqrt(param.h)
        for w in wls:
            w.data.uniform_(-stdv, stdv)

        self.old_state=Variable(torch.Tensor(param.bs,param.h).zero_())


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

        # I think .data call is safe whenever new Variable should be initated.
        # Unless I wish to have the gradients recursively flow back to the beginning of history
        # I do not wish so.
        semicolon_input=torch.cat((input_x,previous_time,previous_layer),dim=1)

        # 5 equations
        input_gate=torch.sigmoid(torch.matmul(semicolon_input, self.W_input)+self.b_input)
        forget_gate=torch.sigmoid(torch.matmul(semicolon_input, self.W_forget)+self.b_forget)
        new_state=forget_gate * self.old_state + input_gate * \
                  torch.tanh(torch.matmul(semicolon_input, self.W_state)+self.b_state)
        output_gate=torch.sigmoid(torch.matmul(semicolon_input, self.W_output)+self.b_output)
        new_hidden=output_gate*torch.tanh(new_state)

        # TODO Warning: needs to assign, not sure if this is right
        # Good warning, I have changed the assignment and I hope this now works better.
        self.old_state=new_state

        return new_hidden

    def new_sequence_reset(self):
        self.old_state=Variable(torch.Tensor(param.bs,param.h).zero_())
        # self.W_input.weight.detach()
        # self.W_input.bias.detach()
        # self.W_forget.weight.detach()
        # self.W_forget.bias.detach()
        # self.W_output.weight.detach()
        # self.W_output.bias.detach()
        # self.W_state.weight.detach()
        # self.W_state.bias.detach()
        # self.old_state.detach()
        self.W_input= Parameter(self.W_input.data,requires_grad=True)
        self.W_forget=Parameter(self.W_forget.data,requires_grad=True)
        self.W_output=Parameter(self.W_output.data,requires_grad=True)
        self.W_state=Parameter(self.W_state.data,requires_grad=True)

        self.b_input= Parameter(self.b_input.data,requires_grad=True)
        self.b_forget=Parameter(self.b_forget.data,requires_grad=True)
        self.b_output=Parameter(self.b_output.data,requires_grad=True)
        self.b_state=Parameter(self.b_state.data,requires_grad=True)

        print('reset controller LSTM')
