from torch.nn.modules import LSTM
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable


class PriorLSTM(nn.Module):
    def __init__(self,prior,input_size=52686, output_size=2976,hidden_size=128,num_layers=16,batch_first=True,
                 dropout=0.1):
        super(PriorLSTM, self).__init__()
        self.lstm=LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
                       batch_first=batch_first,dropout=dropout)
        self.bn = nn.BatchNorm1d(input_size)
        self.output=nn.Linear(hidden_size,output_size)
        self.reset_parameters()
        self.prior=prior

        '''prior'''
        # this is the prior probability of each label predicting true
        # this is added to the logit
        self.prior=prior
        if isinstance(self.prior, np.ndarray):
            self.prior=torch.from_numpy(self.prior).float()
            self.prior=Variable(self.prior, requires_grad=False)
        elif isinstance(self.prior, torch.Tensor):
            self.prior=Variable(self.prior, requires_grad=False)
        else:
            assert(isinstance(self.prior, Variable))


        # transform to logits
        # because we are using sigmoid, not softmax, self.prior=log(P(y))-log(P(not y))
        # sigmoid_input = z + self.prior
        # z = log(P(x|y)) - log(P(x|not y))
        # sigmoid output is the posterior positive
        self.prior=self.prior.clamp(1e-8, 1 - 1e-8)
        self.prior=torch.log(self.prior)-torch.log(1-self.prior)
        a=Variable(torch.Tensor([0]))
        self.prior=torch.cat((a,self.prior))
        self.prior=self.prior.cuda()


        for name, param in self.named_parameters():
            print(name, param.data.shape)

        print("Using prior lstm")

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.output.reset_parameters()

    def forward(self, input, hx=None):
        input=input.permute(0,2,1).contiguous()
        bnout=self.bn(input)
        bnout[(bnout != bnout).detach()] = 0
        input=bnout.permute(0,2,1).contiguous()
        output,statetuple=self.lstm(input,hx)
        output=self.output(output)
        # (batch_size, seq_len, target_dim)
        # pdb.set_trace()
        # output=output.sum(1)
        output=output.max(1)[0]
        output=output+self.prior

        return output
