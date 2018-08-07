import torch
from torch.nn import LSTM
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter


"""
This is an experiement with the detach and save states method referenced in the PyTorch forum.
https://discuss.pytorch.org/t/solved-training-a-simple-rnn/9055/17
https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time-while-using-custom-loss-function/12360
https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/6

This allows me to step through the LSTM and finely control which variable to retain and which to delete.
"""

def main():
    # lstm=LayeredLSTM()
    lstm=LSTM(input_size=47764, hidden_size=128, num_layers=8, batch_first=True)

    # this has no new sequence reset
    # I wonder if gradient information will increase indefinitely

    # Even so, I think detaching at the beginning of each new sequence is an arbitrary decision.
    optim=torch.optim.Adam(lstm.parameters())
    lstm.cuda()
    h0=Variable(torch.rand(8,128,128)).cuda()
    c0=Variable(torch.rand(8,128,128)).cuda()
    states=(h0,c0)
    savedetach=[states]
    sd(savedetach)

    for m in range(1000):
        for _ in range(10):
            print(_)
            scope(optim,lstm,savedetach)
        for i in range(len(savedetach)):
            del savedetach[0]
        h0 = Variable(torch.rand(8, 128, 128)).cuda()
        c0 = Variable(torch.rand(8, 128, 128)).cuda()
        states = (h0, c0)
        savedetach+=[states]

def scope(optim, lstm, savedetach):
    optim.zero_grad()
    input = Variable(torch.rand(128, 1, 47764)).cuda()
    target = Variable(torch.rand(128, 1, 128)).cuda()
    output, states = lstm(input, savedetach[-1])

    savedetach.append(states)
    sd(savedetach)
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(output, target)
    loss.backward()
    optim.step()

def sd(savedetach):
    for s in savedetach[-1]:
        s.detach_()
        s.requires_grad=True

if __name__=="__main__":
    main()