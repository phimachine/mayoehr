from torch.nn.modules import LSTM
import torch.nn as nn

class lstmwrapper(nn.Module):
    def __init__(self,input_size=69505, output_size=5952,hidden_size=128,num_layers=16,batch_first=True,
                 dropout=True):
        super(lstmwrapper, self).__init__()
        self.lstm=LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
                       batch_first=batch_first,dropout=dropout)
        self.output=nn.Linear(hidden_size,output_size)
        self.hx=None
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.output.reset_parameters()

    def assign_state_tuple(self,state_tuple):
        self.hx=state_tuple

    def forward(self, input):
        output,statetuple=self.lstm(input,self.hx)
        return self.output(output), statetuple
