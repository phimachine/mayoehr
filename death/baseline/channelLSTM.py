from torch.nn.modules import LSTM
import torch.nn as nn
from torch.autograd import Variable
import torch
import pdb

class ChannelLSTM(nn.Module):
    def __init__(self,input_size=69505, output_size=5952,hidden_size=52,num_layers=16,batch_first=True,
                 dropout=True):
        super(ChannelLSTM, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
                       batch_first=batch_first,dropout=dropout)
        self.output=nn.Linear(self.hidden_size,output_size)
        self.hx=None
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.output.reset_parameters()

    def init_states_each_channel(self):
        # num_layers, channel, dim
        h=Variable(torch.Tensor(self.num_layers, 1, self.hidden_size)).cuda().zero_()
        s=Variable(torch.Tensor(self.num_layers, 1, self.hidden_size)).cuda().zero_()
        return (h,s)

    def assign_states_tuple(self, states_tuple):
        self.hx=states_tuple

    def forward(self, input):
        output,statetuple=self.lstm(input,self.hx)
        output=output.squeeze(1)
        output=self.output(output)
        return output, statetuple
#
# class ChannelLSTM2(nn.Module):
#     def __init__(self,input_size=69505, output_size=5952,hidden_size=128,num_layers=32,batch_first=True,
#                  dropout=True):
#         super(ChannelLSTM2, self).__init__()
#         self.hidden_size=hidden_size
#         self.num_layers=num_layers
#         self.lstm=LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
#                        batch_first=batch_first,dropout=dropout)
#         self.output=nn.Linear(self.hidden_size,output_size)
#         self.hx=None
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.lstm.reset_parameters()
#         self.output.reset_parameters()
#
#     def init_states_each_channel(self):
#         # num_layers, channel, dim
#         h=Variable(torch.Tensor(self.num_layers, 1, self.hidden_size)).cuda().zero_()
#         s=Variable(torch.Tensor(self.num_layers, 1, self.hidden_size)).cuda().zero_()
#         return (h,s)
#
#     def assign_states_tuple(self, states_tuple):
#         self.hx=states_tuple
#
#     def forward(self, input):
#         output,statetuple=self.lstm(input,self.hx)
#         output=output.squeeze(1)
#         output=self.output(output)
#         return output, statetuple
#
#
#
# class Custom_LSTM(nn.Module):
#
#     def __init__(self,input_size=69505, output_size=5952,hidden_size=128,num_layers=32):
#         super(Custom_LSTM, self).__init__()
#
#
# class LSTM_Unit(nn.Module):
#     """
#     A single layer unit of LSTM
#     """
#
#     def __init__(self, x, R, W, h, bs):
#         super(LSTM_Unit, self).__init__()
#
#         self.x = x
#         self.R = R
#         self.W = W
#         self.h = h
#         self.bs = bs
#
#         self.W_input = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
#         self.W_forget = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
#         self.W_output = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
#         self.W_state = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
#
#         self.old_state = Variable(torch.Tensor(self.bs, self.h).zero_().cuda(),requires_grad=False)
#
#     def reset_parameters(self):
#         for module in self.children():
#             module.reset_parameters()
#
#     def forward(self, input_x, previous_time, previous_layer):
#         # a hidden unit outputs a hidden output new_hidden.
#         # state also changes, but it's hidden inside a hidden unit.
#
#         semicolon_input = torch.cat([input_x, previous_time, previous_layer], dim=1)
#
#         # 5 equations
#         input_gate = torch.sigmoid(self.W_input(semicolon_input))
#         forget_gate = torch.sigmoid(self.W_forget(semicolon_input))
#         new_state = forget_gate * self.old_state + input_gate * \
#                     torch.tanh(self.W_state(semicolon_input))
#         output_gate = torch.sigmoid(self.W_output(semicolon_input))
#         new_hidden = output_gate * torch.tanh(new_state)
#         self.old_state = Parameter(new_state.data,requires_grad=False)
#
#         return new_hidden
#
#
#     def reset_batch_channel(self,list_of_channels):
#         raise NotImplementedError()
#     #
#     # def new_sequence_reset(self):
#     #     raise DeprecationWarning("We no longer reset sequence together in all batch channels, this function deprecated")
#     #
#     #     self.W_input.weight.detach()
#     #     self.W_input.bias.detach()
#     #     self.W_output.weight.detach()
#     #     self.W_output.bias.detach()
#     #     self.W_forget.weight.detach()
#     #     self.W_forget.bias.detach()
#     #     self.W_state.weight.detach()
#     #     self.W_state.bias.detach()
#     #
#     #     self.old_state = Parameter(torch.Tensor(self.bs, self.h).zero_().cuda(),requires_grad=False)
