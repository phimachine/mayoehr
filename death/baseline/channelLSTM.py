from torch.nn.modules import LSTM
import torch.nn as nn
from torch.autograd import Variable
import torch
import pdb

class channelLSTM(nn.Module):
    def __init__(self,input_size=69505, output_size=5952,hidden_size=128,num_layers=32,batch_first=True,
                 dropout=True):
        super(channelLSTM, self).__init__()
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
