from death.post.inputgen_planC import *


# we can only assume that all deaths are recorded

# torch integration reference: https://github.com/utkuozbulak/pytorch-custom-dataset-examples

# This is the plan C.

def get_timestep_location(earliest, dates):
    """
    Uses numpy instead of panda.
    dates must be month based, otherwise errors will arise.

    :param earliest: pandas.Timestamp
    :param dates: ndarray of datetime64
    :return: cc: int numpy array, the index location of the corresponding records
    """
    dates = dates.apply(lambda x: x.replace(day=1))
    earliest = earliest.to_datetime64()
    # if not isinstance(time,pd.Timestamp):
    # # if it's a series as it should be
    #     time=time.values
    # else:
    #     time=time.to_datetime64()
    dates = dates.values
    cc = (dates - earliest).astype('timedelta64[M]')
    return cc.astype("int")


class InputGenD(InputGen):
    # inherit this class mainly so that I can reuse the code and maintain faster
    def __init__(self, death_proportion=0.5, load_pickle=True, verbose=False, debug=False):
        super(InputGenD, self).__init__(load_pickle=load_pickle, verbose=verbose, debug=debug)
        # This is sorted
        death_rep_person_id = self.death.index.get_level_values(0).unique().values
        no_death_rep_person_id = self.rep_person_id[np.invert(np.in1d(self.rep_person_id, death_rep_person_id))]
        self.death_proportion = death_proportion
        leng = int(len(death_rep_person_id) / death_proportion)
        no_death_rep_person_id = np.random.choice(no_death_rep_person_id, size=leng - len(death_rep_person_id),
                                                  replace=False)
        self.all_indices = np.concatenate((death_rep_person_id, no_death_rep_person_id))
        np.random.shuffle(self.all_indices)
        self.len = len(self.all_indices)
        print("InputGenD finished")

    def __getitem__(self, index, debug=False):
        # TODO there is a critical bug that shows that my post processing might need to be updated.
        # TODO somehow a value that is in rep_person_id+death is not appearing in earla, which means
        # TODO it does not have a record in our db.
        # index=28119
        # id=99102
        id = self.all_indices[index]
        return self.get_by_id(id, debug)

    def __len__(self):
        return self.len


class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10, random_seed=None):
    """
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    This is just a factory method, nothing special. Can't believe no one ever did this for PyTorch.

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


def oldmain():
    ig = InputGen(load_pickle=True, verbose=False)
    ig.performance_probe()

    # go get one of the values and see if you can trace it all the way back to raw data
    # this is a MUST DO TODO
    print('parallelize')
    dl = DataLoader(dataset=ig, batch_size=1, shuffle=False, num_workers=16)

    start = time.time()
    n = 0
    for i, t, l, in dl:
        print(i, t, l)
        # if you take a look at the shape you will know that dl expanded a batch dim
        n += 1
        if n == 100:
            break
        if (i != i).any() or (t != t).any():
            raise ValueError("NA found")
    end = time.time()
    print("100 rounds, including initiation:", end - start)
    # batch data loading seems to be a problem since patients have different lenghts of data.
    # it's advisable to load one at a time.
    # we need to think about how to make batch processing possible.
    # or maybe not, if the input dimension is so high.
    # well, if we don't have batch, then we don't have batch normalization.
    print("script finished")


if __name__ == "__main__":
    ig = InputGenD(load_pickle=True, verbose=False)
    train, valid = train_valid_split(ig)
    traindl = DataLoader(dataset=train, batch_size=1)
    validdl = DataLoader(dataset=valid, batch_size=1)
    print(train[1])
    for x, y in enumerate(traindl):
        if x == 2:
            break
        print(y)
    for x, y in enumerate(validdl):
        if x == 2:
            break
        print(y)

    for x, y in enumerate(traindl):
        if x == 100:
            break
        print(y[2])
