import torch.nn as nn
import pickle
from torch.nn.modules import LSTM
from torch.autograd import Variable
import torch

class Stock_LSTM(nn.Module):
    """
    I prefer using this Stock LSTM for numerical stability.
    """
    def __init__(self, x, R, W, h, L, v_t):
        super(Stock_LSTM, self).__init__()

        self.x = x
        self.R = R
        self.W = W
        self.h = h
        self.L = L
        self.v_t= v_t

        self.LSTM=LSTM(input_size=self.x+self.R*self.W,hidden_size=h,num_layers=L,batch_first=True,
                       dropout=True)
        self.last=nn.Linear(self.h, self.v_t)
        self.st=None

    def forward(self, input_x):
        """
        :param input_x: input and memory values
        :return:
        """
        assert (self.st is not None)
        o, st = self.LSTM(input_x, self.st)
        if (st[0]!=st[0]).any():
            with open("debug/lstm.pkl") as f:
                pickle.dump(self, f)
            with open("debug/lstm.pkl") as f:
                pickle.dump(input_x, f)
            raise ("LSTM produced a NAN, objects dumped.")
        return self.last(o), st

    def reset_parameters(self):
        self.LSTM.reset_parameters()
        self.last.reset_parameters()

    def assign_states_tuple(self, states_tuple):
        self.st=states_tuple


def testbid():
    lstm=LSTM(input_size=100, hidden_size=77, batch_first=True, dropout=True)
    lstmbi=LSTM(input_size=100, hidden_size=77, batch_first=True, dropout=True, bidirectional=True)
    input=Variable(torch.Tensor(64,8,100))
    output=lstm(input, None)
    outputbi=lstmbi(input, None)
    print(output[0].shape, output[1][0].shape, output[1][1].shape)
    print(outputbi[0].shape, outputbi[1][0].shape, output[1][1].shape)
    print("done")

if __name__ == '__main__':
    testbid()