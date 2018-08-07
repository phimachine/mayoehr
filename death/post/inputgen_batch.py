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