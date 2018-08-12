from torch.autograd import Variable
import torch

debug=True

class Channel():

    def __init__(self):
        # you need to call set_next_sequences
        # a channel itself does not have any sequence specific property
        super(Channel, self).__init__()
        self.saved_states = []
        self.current_sequences = None
        self.current_seq_len = None
        self.current_step = None
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
        timestep_feed = list(torch.index_select(seq, 1, torch.LongTensor([self.current_step]))
                         for seq in self.current_sequences)
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
