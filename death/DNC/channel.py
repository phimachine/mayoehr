from torch.autograd import Variable
import torch


class Channel():

    def __init__(self):
        super(Channel, self).__init__()
        self.saved_states = []
        self.current_sequences = None
        self.current_seq_len = None
        self.current_step = None
        self.next_feed = None
        self.static_feed = None

    def set_next_sequences(self, next_sequences, static_feed):
        self.current_sequences = next_sequences
        # assume that time dimension is 1
        self.current_seq_len = self.current_sequences[0].shape[1]
        self.current_step = 0
        self.static_feed = static_feed

    def reinit_states(self):
        # you can set the next sequence early, but you cannot clean up before the loss.backward() has been called.
        for i in range(len(self.saved_states)):
            del self.saved_states[0]
        self.init_states()

    def init_states(self):
        h0 = Variable(torch.rand(8, 1, 512)).cuda()
        c0 = Variable(torch.rand(8, 1, 512)).cuda()
        states = (h0, c0)
        self.saved_states = [states]

    def step(self):
        # this is the main function you should call.
        # this function will return all the correct inputs and communicate with the channel manager to request new
        # data.

        # we call clean up before the next input is used, after the last loss.backward() has been called
        new_sequence_request = False
        if self.current_step == 0:
            self.reinit_states()

        # assume time dimension is 1
        timestep_feed = (seq.index_select(1, Variable(torch.cuda.LongTensor(current_step))) for seq in self.current_sequences)
        self.current_step += 1

        # assert that there is no sequence of length 0
        if self.current_step == self.current_seq_len:
            new_sequence_request = True
        return timestep_feed, self.static_feed, new_sequence_request

    def get_states(self):
        return self.saved_states[-1]

    def set_states(self, states):
        self.saved_states.append(states)
        for s in self.saved_states[-1]:
            s.detach_()
            s.requires_grad = True


class InputGenBatch():
    """
    The channel manager keeps track of all sequences' remaining length.
    """

    def __init__(self, dataloader, batch_size, dlbatchdims):
        super(ChannelManager, self).__init__()
        self.channels = []
        self.bs = batch_size
        self.dataloader = dataloader
        self.dliter = iter(self.dataloader)
        # the dimensions of dataloader yield that will need to made batch
        self.dlbatchdims = dlbatchdims
        self.dltuplelen=None

        self._add_channels(self.bs)
        for ch in self.channels:
            new_sequences, static_feed=self.get_new_sequences()
            if self.dltuplelen is None:
                self.dltuplelen = len(newdata)
            if debug:
                for tensor in new_sequences:
                    assert self.lengths[-1] == tensor.shape[1]
            ch.set_next_sequences(new_sequences, static_feed)

    def _add_channels(self, num):
        for i in range(num):
            self.channels.append(Channel())

    def get_new_sequences(self):
        newdata = next(self.dliter)
        new_sequences = (newdata[i] for i in self.dlbatchdims)
        static_feed = (newdata[i] for i in range(self.dltuplelen) if i not in self.dlbatchdims)
        return new_sequences, static_feed

    def __next__(self):
        try:
            # not using cat call because the step has a reset flag.

            batch_timestep_feed=[]
            batch_static_feed=[]
            for ch in self.channels:
                tf, sf, new= ch.step()
                batch_timestep_feed.append(tf)
                batch_static_feed.append(sf)
                if new:
                    ch.set_next_sequences(*self.get_new_sequences())
                    # at this step, the states are not cleaned, so backprop can happen.
            # assume input, target both have batch to be the first dimension
            return torch.cat(batch_timestep_feed,0), torch.cat(batch_static_feed,0)
        except StopIteration:
            raise StopIteration()

    def cat_call(self, func_name, dim=0):
        # the return can be (Tensor, Tensor)

        res = []
        noret = False
        for ch in self.channels:
            func = getattr(ch, func_name)
            ret = func()
            if ret is None:
                noret = True
            else:
                res.append(ret)
        if noret:
            return
        else:
            if isinstance(res[0], torch.Tensor) or isinstance(res[0], torch.autograd.Variable):
                return torch.cat(res, dim)
            else:
                unzipped = list(zip(*res))
                return tuple(torch.cat(m, dim) for m in unzipped)

    def distribute_call(self, func_name, arg, dim=0):
        try:
            for i in range(self.bs):
                func = getattr(self.channels[i], func_name)
                func(arg.index_select(dim, i))
        except AttributeError:
            for i in range(self.bs):
                var = Variable(torch.cuda.LongTensor([i]))
                call_tuple = [tensor.index_select(dim, var) for tensor in arg]
                func = getattr(self.channels[i], func_name)
                func(call_tuple)

    def get_channel(self, index):
        return self.channels[index]
