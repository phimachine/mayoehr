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


def sd(savedetach):
    # this is the critical function.
    for s in savedetach[-1]:
        s.detach_()
        s.requires_grad = True


class Channel():
    def __init__(self):
        super(Channel, self).__init__()
        self.saved_states = []

    def new_sequence(self):
        for i in range(len(self.saved_states)):
            del self.saved_states[0]
        self.init_states()

    def init_states(self):
        h0 = Variable(torch.rand(8, 1, 128)).cuda()
        c0 = Variable(torch.rand(8, 1, 128)).cuda()
        states = (h0, c0)
        self.saved_states = [states]

    def get_input(self):
        return Variable(torch.rand(1, 1, 47764)).cuda()

    def get_states(self):
        return self.saved_states[-1]

    def push_states(self,states):
        self.saved_states.append(states)
        for s in self.saved_states[-1]:
            s.detach_()
            s.requires_grad=True


class ChannelManager():
    def __init__(self):
        super(ChannelManager, self).__init__()
        self.channels = []
        self.bs=0

    def add_channels(self, num):
        for i in range(num):
            self.channels.append(Channel())

    def cat_call(self, func_name, dim=0):
        # the return can be (Tensor, Tensor)

        res = []
        noret=False
        for ch in self.channels:
            func = getattr(ch, func_name)
            ret=func()
            if ret is None:
                noret=True
            else:
                res.append(ret)
        if noret:
            return
        else:
            if isinstance(res[0],torch.Tensor) or isinstance(res[0], torch.autograd.Variable):
                return torch.cat(res,dim)
            else:
                unzipped=list(zip(*res))
                return tuple(torch.cat(m, dim) for m in unzipped)

    def distribute_call(self,func_name,arg):
        try:
            for i in range(self.bs):
                func=getattr(self.channels[i],func_name)
                func(arg.index_select(0,i))
        except AttributeError:
            for i in range(self.bs):
                call_tuple=[tensor.index_select(0,i) for tensor in arg]
                func = getattr(self.channels[i], func_name)
                func(call_tuple)

    def __getitem__(self, item):
        return self.channels[item]


def main0():
    # lstm=LayeredLSTM()
    lstm = LSTM(input_size=47764, hidden_size=128, num_layers=8, batch_first=True)

    # this has no new sequence reset
    # I wonder if gradient information will increase indefinitely

    # Even so, I think detaching at the beginning of each new sequence is an arbitrary decision.
    optim = torch.optim.Adam(lstm.parameters())
    lstm.cuda()
    h0 = Variable(torch.rand(8, 2, 128)).cuda()
    c0 = Variable(torch.rand(8, 2, 128)).cuda()
    states = (h0, c0)
    savedetach = [states]

    sd(savedetach)

    for m in range(1000):
        for _ in range(10):
            print(_)
            optim.zero_grad()
            input = Variable(torch.rand(2, 1, 47764)).cuda()
            target = Variable(torch.rand(2, 1, 128)).cuda()
            output, states = lstm(input, savedetach[-1])

            savedetach.append(states)
            sd(savedetach)
            criterion = torch.nn.SmoothL1Loss()
            loss = criterion(output, target)
            loss.backward()
            optim.step()

        for i in range(len(savedetach)):
            del savedetach[0]
        h0 = Variable(torch.rand(8, 128, 128)).cuda()
        c0 = Variable(torch.rand(8, 128, 128)).cuda()
        states = (h0, c0)
        savedetach += [states]

def main():
    # lstm=LayeredLSTM()
    lstm = LSTM(input_size=47764, hidden_size=128, num_layers=8, batch_first=True)

    # this has no new sequence reset
    # I wonder if gradient information will increase indefinitely

    # Even so, I think detaching at the beginning of each new sequence is an arbitrary decision.
    optim = torch.optim.Adam(lstm.parameters())
    lstm.cuda()
    criterion = torch.nn.SmoothL1Loss()

    cm=ChannelManager()
    cm.add_channels(2)
    cm.cat_call("init_states")

    for i in range(1000):
        print(i)
        optim.zero_grad()
        target = Variable(torch.rand(2,1,128)).cuda()
        output, states= lstm(cm.cat_call("get_input"), cm.cat_call("get_states", 1))
        cm.distribute_call("push_states",states)
        loss = criterion(output, target)
        loss.backward()
        optim.step()

        if i % 3 == 0:
            cm[0].new_sequence()


if __name__ ==  "__main__":
    main()
