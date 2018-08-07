"""
D is different from C because it rebalances the dataset to have more death records, so that the optimal
prediction strategy is not to output zero for all.

E is different in the sense that it outputs a timepoint, instead of a series.
Note that the model will take not just the input but also the hidden and state for LSTM,
all memory configuration for DNC. These inputs are not supplied by the training loop, instead of the
DataLoader. This should work.

Note that this inputgen only works for the palette plan, which resets experience for each new sequence.
"""

'''
Design plan:
A parallelized worker pool supplies sequences of patient health records with shuffled order and 
death proportion adjusted.
A Splitter takes all the sequences and open batch_num channels of outputs. It caches sequences and 
concatenate them to whichever channel that runs out of patient records. This object that does deal
with model states. Model states are cached in the function scope.
Splitter object is not a Dataset, because we do not define a __len__() method on Splitter.
'''
from death.post.inputgen_planD import *
from death.DNC.channel import *

debug=True


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
    def __init__(self,dataloader, batch_size):
        super(ChannelManager, self).__init__()
        self.channels = []
        self.bs=batch_size
        self.dataloader=dataloader

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


class InputGenBatch():
    '''
    This is the channel based object that produces time-indexed values.
    This is built to be an iterator.
    '''
    def __init__(self, batch_size, dataloader, dldims=(0,1), tensor_time_dim=(1,1)):
        """
        :param batch_size:
        :param dataloader:
        :param dldims: list of dataloader yield values that need to be processed.
                       e.g. if yield is (input, target, loss_type), then dldims=(0,1)
        :param time_seq_dim: the dimension of the tensor
        """

        self.batch_size=batch_size
        self.dataloader=dataloader
        self.dliter=iter(self.dataloader)
        self.dldims=dldims
        self.tensor_time_dim=tensor_time_dim

        # initialize all
        # all the sequences for the next yield value.
        self.channels=[next(self.dliter) for _ in range(self.batch_size)]
        # all timestep values for the current yield values.
        # index pointer for all sequences, we assume all dataloader tensors have same timesteps.
        self.timesteps= [0] * batch_size
        # length of current sequence
        self.len=[]
        for ch in self.channels:
            self.len+=[ch[self.dldims[0]].shape[tensor_time_dim[0]]]
            if debug:
                for ti,ttd in zip(self.dldims,self.tensor_time_dim):
                    tensor=ch[ti]
                    assert self.len[-1]==tensor.shape[ttd]

        self.dltuplelen=len(self.channels[0])

    def __iter__(self):
        return self

    def __next__(self):
        """

        :return: batch for those that need batches. the last one is a list of reset signals.
        """
        try:
            # we should not use tensor here. There is too much logic.
            ret=[]
            for i in range(self.dltuplelen):
                if i in self.dldims:
                    ret.append([])
                else:
                    ret.append(torch.stack([ch[i] for ch in self.channels]))
            for channel, timestep in zip(self.channels, self.timesteps):
                for i,dldim in enumerate(self.dldims):
                    ret[dldim]+=[channel[dldim].index_select(self.tensor_time_dim[i],torch.LongTensor([timestep]))]
            for i in self.dldims:
                ret[i]=torch.cat(ret[i],dim=0)
            for i in range(self.batch_size):
                self.timesteps[i]= self.timesteps[i] + 1
                if self.timesteps[i]==self.len[i]:
                    # require a new sequence on channel i
                    ch=next(self.dliter)
                    self.channels[i]=ch
                    self.len[i]=ch[self.dldims[0]].shape[self.tensor_time_dim[0]]
                    self.timesteps[i]=0

            reset_state=[i==0 for i in self.timesteps]
            ret.append(reset_state)
            return ret

        except StopIteration: # this exception will be raised when calling next(self.dliter)
            # StopIteration will occur for the shortest channel. Discarding rest of the channels.
            raise StopIteration()

def train_valid_split(ds, split_fold=10, random_seed=12345):
    """
    This is a pytorch generic function that takes a data. Dataset object and splits it to validation and training
    efficiently.
    This is just a factory method, nothing special. Can't believe no one ever did this for PyTorch.

    You need to fix the seed so that when this object is reinitiated, no valid leaks into train.

    :return:
    """

    if random_seed is None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    valid_size = dslen // split_fold
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)

    return train, valid

def main():
    ig=InputGenD(load_pickle=True, verbose=False)
    train, valid= train_valid_split(ig)
    traindl= DataLoader(dataset=train,batch_size=1)
    bige=InputGenBatch(16,traindl)
    print(next(bige))

if __name__=="__main__":
    main()