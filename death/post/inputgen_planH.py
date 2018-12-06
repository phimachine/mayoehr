# Plan H caches InputGenG
# instead of loading pandas pickles, it loads numpy pickles directly.
# I traded 1TB storage space for computation speed. This is worth it definitely.
# InputGenG will still be available in the object. However, the heavy lifting work is not done by its __getitem__()
# InputGenH is very rigid. It does not allow any __init__() parameters, because almost any change would require
# recaching the whole InputGenG dataset.

from death.post.dfmanager import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from numpy.random import permutation
import time
import torch
from multiprocessing.pool import ThreadPool as Pool
from death.post.inputgen_planG import *

class InputGenH():

    def __init__(self):
        self.dspath="/infodev1/rep/projects/jason/cache/"
        self.inputgenG=InputGenG(death_fold=0)

    def get_valid(self):
        return DatasetCacher(id="valid", dataset=self.inputgenG.get_valid(), path=self.dspath)

    def get_test(self):
        return DatasetCacher(id="test", dataset=self.inputgenG.get_test(), path=self.dspath)

    def get_train(self):
        return DatasetCacher(id="train", dataset=self.inputgenG.get_train(), path=self.dspath)



class DatasetCacher(Dataset):

    def __init__(self,id,dataset,path="/infodev1/rep/projects/jason/cache/"):
        self.path=path
        self.dataset=dataset
        self.id=id

    def cache_one(self,index):
        point = self.dataset[index]
        fname = self.path + self.id + "_" + str(index) + ".pkl"
        with open(fname, "wb+") as f:
            pickle.dump(point,f,protocol=pickle.HIGHEST_PROTOCOL)

    def cache_all(self,num_workers):
        # cache does not cache the whole dataset object
        # it caches the __getitem__ method only

        dataset_len=len(self.dataset)
        pool=Pool(num_workers)
        for i in range(dataset_len):
            pool.apply_async(self.cache_one, (i,))
        pool.close()
        pool.join()

    def __getitem__(self, index):
        fname = self.path + self.id + "_" + str(index) + ".pkl"
        with open(fname, "rb") as f:
            item=pickle.load(f)
        return item

    def __len__(self):
        return len(self.dataset)

def main():
    ig=InputGenH()
    valid=ig.get_valid()
    # train=ig.get_train()
    # print(valid[100])
    # print(train[192])
    cacher=DatasetCacher("test/valid",valid)
    # cacher.cache_one(55)
    # print(cacher[55])
    # cacher.cache_all(2)
    # print("Done")
    print(cacher[2])

if __name__=="__main__":
    main()