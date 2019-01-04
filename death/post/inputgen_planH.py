# Plan H caches InputGenG
# instead of loading pandas pickles, it loads numpy pickles directly.
# I traded 1TB storage space for computation speed. This is worth it definitely.
# InputGenG will still be available in the object. However, the heavy lifting work is not done by its __getitem__()
# InputGenH is very rigid. It does not allow any __init__() parameters, because almost any change would require
# recaching the whole InputGenG dataset.

# 1/2
# underlying label

from death.post.inputgen_planG import *

class DatasetCacher(Dataset):

    def __init__(self,id,dataset,len=None,max=None,path="/local2/tmp/jasondata/",small_target=False):
        self.path=path
        self.dataset=dataset
        self.id=id
        self.max=max
        self.len=len
        self.small_target=small_target

    def cache_one(self,index):
        pickle_fname = self.path + self.id + "_" + str(index) + ".pkl"
        # if the file does not exist or the file is extremely small
        if not os.path.isfile(pickle_fname) or os.stat(pickle_fname).st_size<16:
            item = self.dataset[index]
            with open(pickle_fname,"wb+") as f:
                pickle.dump(item,f)
            return item
        else:
            fname = self.path + self.id + "_" + str(index) + ".pkl"
            try:
                with open(fname, "rb") as f:
                    item = pickle.load(f)
                return item
            except FileNotFoundError:
                traceback.print_exc()
                print("This line should never be reached")

    def cache_all(self,num_workers):
        # cache does not cache the whole dataset object
        # it caches the __getitem__ method only

        dataset_len=len(self.dataset)
        pool=Pool(num_workers)
        for i in range(dataset_len):
            pool.apply_async(self.cache_one, (i,))
        pool.close()
        pool.join()

    def cache_some(self,num_workers):
        # cache does not cache the whole dataset object
        # it caches the __getitem__ method only
        assert (self.max is not None)
        dataset_len=len(self.dataset)
        if self.max >dataset_len:
            self.max=dataset_len
        pool=Pool(num_workers)
        for i in range(self.max):
            pool.apply_async(self.cache_one, (i,))
        pool.close()
        pool.join()

    # def __getitem__(self, index):
    #
    #     if self.max is None or index<self.max:
    #         fname = self.path + self.id + "_" + str(index) + ".pkl"
    #         try:
    #             with open(fname, "rb") as f:
    #                 item=pickle.load(f)
    #             return item
    #         except FileNotFoundError:
    #             return self.cache_one(index)
    #     else:
    #         return self.cache_one(index)

    def __getitem__(self, index):

        fname = self.path + self.id + "_" + str(index) + ".pkl"
        try:
            with open(fname, "rb") as f:
                item=pickle.load(f)
            if self.small_target:
                st=item[1][:2976]
                return item[0], st, item[2]
            else:
                return item
        except FileNotFoundError:
            raise

    def __len__(self):
        if self.len is None:
            return len(self.dataset)
        else:
            return self.len


def cache_them():
    ig=InputGenG(death_fold=0)
    ig.train_valid_test_split()
    valid=ig.get_valid()
    valid_cacher=DatasetCacher("zerofold/valid",valid)
    train=ig.get_train()
    train_cacher=DatasetCacher("zerofold/train",train)
    test=ig.get_test()
    test_cacher=DatasetCacher("zerofold/test",test)

    valid_cacher.cache_all(16)
    train_cacher.cache_all(16)
    test_cacher.cache_all(16)

    print("Done")

def selective_cache():
    ig=InputGenG(death_fold=0)
    ig.train_valid_test_split()
    train=ig.get_train()
    train_cacher=DatasetCacher("zerofold/train",train,max=50000)
    # train_cacher.cache_one(2)
    train_cacher.cache_some(16)
    valid=ig.get_valid()
    valid_cacher=DatasetCacher("zerofold/valid",valid, max=5000)
    valid_cacher.cache_some(16)

    test=ig.get_test()
    test_cacher=DatasetCacher("zerofold/test",test, max=5000)
    test_cacher.cache_some(16)
    print("Done")

class InputGenH():

    def __init__(self,small_target=False):
        self.dspath="/local2/tmp/jasondata/zerofold/"
        self.inputgenG=None
        self.small_target=small_target
        print("H initiated")

    def get_valid(self):
        return DatasetCacher(id="valid", dataset=None, len=24742, path=self.dspath, max=5000,small_target=self.small_target)

    def get_test(self):
        return DatasetCacher(id="test", dataset=None, len=24742, path=self.dspath,max=5000,small_target=self.small_target)

    def get_train(self):
        return DatasetCacher(id="train", dataset=None, len=197944, path=self.dspath, max=50000,small_target=self.small_target)
#
#
# class InputGenH():
#
#     def __init__(self):
#         self.dspath="/local2/tmp/jasondata/zerofold/"
#         self.inputgenG=InputGenG(death_fold=0)
#         print("G initiated")
#
#     def get_valid(self):
#         return DatasetCacher(id="valid", dataset=self.inputgenG.get_valid(), path=self.dspath, max=5000)
#
#     def get_test(self):
#         return DatasetCacher(id="test", dataset=self.inputgenG.get_test(), path=self.dspath,max=5000)
#
#     def get_train(self):
#         return DatasetCacher(id="train", dataset=self.inputgenG.get_train(), path=self.dspath, max=50000)


def target_investigation():
    ig=InputGenH()
    valid=ig.get_valid()
    for i in range(100):
        input, target, loss_type=valid[i]
        print(target[0])

def main():
    ig=InputGenH()
    valid=ig.get_valid()
    print(valid[4])

if __name__=="__main__":
    target_investigation()