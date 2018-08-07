from torch.autograd import Variable
import torch


class Channel():
    def __init__(self):
        super(Channel, self).__init__()
        self.saved_states = []

    def new_sequence(self):
        for i in range(len(self.saved_states)):
            del self.saved_states[0]
        self.init_states()

    def init_states(self):
        h0 = Variable(torch.rand(8, 1, 512)).cuda()
        c0 = Variable(torch.rand(8, 1, 512)).cuda()
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
