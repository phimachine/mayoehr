"""
D is different from C because it rebalances the dataset to have more death records, so that the optimal
prediction strategy is not to output zero for all.

BatchChannel does not modify the input generation plan from D. It takes a dataloader object, maybe Plan D, then
opens several channels and allows you to access the channels step by step. Most importantly, you can pass your graph
states back to this BatchChannel object to feed it on the next timestep, and all channels will handle memory constraints
independently. It will guarantee that backprop will always have all needed states. It will also guarantee that memory
will be freed once no previous states will be needed (at the end of the sequence for a channel).
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


class BatchChannel():
    """
    The channel manager keeps track of all sequences' remaining length.
    """

    def __init__(self, dataloader, batch_size, seqdims=(0, 1), staticdims=(2,)):
        super(BatchChannel, self).__init__()
        self.channels = []
        self.bs = batch_size
        self.dataloader = dataloader
        self.dliter = iter(self.dataloader)
        # the dimensions of dataloader yield that will need to made batch
        self.seqdims=seqdims
        self.staticdims=staticdims

        self._add_channels(self.bs)
        for ch in self.channels:
            new_sequences, static_feed=self.get_new_sequences()
            ch.set_next_sequences(new_sequences, static_feed)
            if debug:
                for tensor in new_sequences:
                    assert ch.current_seq_len == tensor.shape[1]

    def _add_channels(self, num):
        for i in range(num):
            self.channels.append(Channel())

    def get_new_sequences(self):
        newdata = next(self.dliter)
        new_sequences = list(newdata[i] for i in self.seqdims)
        static_feed = list(newdata[i] for i in self.staticdims)
        return new_sequences, static_feed

    def __next__(self):
        try:
            # not using cat call because the step has a reset flag.
            batch_seq_feed=[]
            batch_static_feed=[]
            for ch in self.channels:
                tf, sf, new= ch.step()
                batch_seq_feed.append(tf)
                batch_static_feed.append(sf)
                if new:
                    ch.set_next_sequences(*self.get_new_sequences())
                    # at this step, the states are not cleaned, so backprop can happen.
            # assume input, target both have batch to be the first dimension
            batch_seq_feed=list(zip(*batch_seq_feed))
            batch_static_feed=list(zip(*batch_static_feed))
            return [torch.cat(seq,0) for seq in batch_seq_feed], \
                   [torch.cat(static,0) for static in batch_static_feed]
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


# class InputGenBatch():
#     '''
#     This is the channel based input generation that produces time-indexed values.
#     This is built to be an iterator.
#     '''
#     def __init__(self, batch_size, dataloader, seqdims=(0, 1), staticdims=(2,), tensor_time_dim=(1, 1)):
#         """
#         :param batch_size:
#         :param dataloader:
#         :param seqdims: list of dataloader yield values that need to be processed.
#                        e.g. if yield is (input, target, loss_type), then dldims=(0,1)
#         :param staticdims: (cont'd) and the staticdims=(2,)
#         :param time_seq_dim: the dimension of the tensor
#         """
#
#         self.batch_size=batch_size
#         self.dataloader=dataloader
#         self.dliter=iter(self.dataloader)
#         self.seqdims=seqdims
#         self.staticdims=staticdims
#         self.tensor_time_dim=tensor_time_dim
#
#         # initialize all
#         # all the sequences for the next yield value.
#         self.channels=[next(self.dliter) for _ in range(self.batch_size)]
#
#
#         self.dltuplelen=len(self.channels[0])
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         """
#
#         :return: batch for those that need batches. the last one is a list of reset signals.
#         """
#         try:
#             # we should not use tensor here. There is too much logic.
#             ret=[]
#             for i in range(self.dltuplelen):
#                 if i in self.seqdims:
#                     ret.append([])
#                 else:
#                     ret.append(torch.stack([ch[i] for ch in self.channels]))
#             for channel, timestep in zip(self.channels, self.timesteps):
#                 for i,dldim in enumerate(self.seqdims):
#                     ret[dldim]+=[channel[dldim].index_select(self.tensor_time_dim[i],torch.LongTensor([timestep]))]
#             for i in self.seqdims:
#                 ret[i]=torch.cat(ret[i],dim=0)
#             for i in range(self.batch_size):
#                 self.timesteps[i]= self.timesteps[i] + 1
#                 if self.timesteps[i]==self.len[i]:
#                     # require a new sequence on channel i
#                     ch=next(self.dliter)
#                     self.channels[i]=ch
#                     self.len[i]=ch[self.seqdims[0]].shape[self.tensor_time_dim[0]]
#                     self.timesteps[i]=0
#
#             reset_state=[i==0 for i in self.timesteps]
#             ret.append(reset_state)
#             return ret
#
#         except StopIteration: # this exception will be raised when calling next(self.dliter)
#             # StopIteration will occur for the shortest channel. Discarding rest of the channels.
#             raise StopIteration()

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
    traindl= DataLoader(num_workers=8,dataset=train,batch_size=1)
    igb=BatchChannel(traindl, 16)
    for _ in range(1000):
        a=next(igb)


if __name__=="__main__":
    main()