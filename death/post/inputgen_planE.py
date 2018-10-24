# We are now wrapping up the project.
# A problem that Che pointed out is that we are not supposed to modify the validation metrics.
# I think this is a valid concern, and I will rewrite inputgen_planD so that the resizing happens after the
# train and valid split.

# The goal of plan E is to achieve online modification of death proportion.
# Plan:
# 1, Do the validation training split, ensure validation does not leak. This validation set will be the
#    same throughout the whole curriculum.
# 2, Produce a list of no death rep person id for training set with specified death proportion.
#    Concatenate the death and no death.
#    Connect __getitem__() with the big list.
#    This design ensures that in a single epoch, no duplicate training points will be encountered.
#    Not sure if it matters.
# 3, Change death proportion and go back to 2.

from death.post.inputgen_planC import InputGen
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from numpy.random import permutation

class GenHelper(Dataset):
    def __init__(self, ids, ig):
        """

        :param ids: needs to be combined and randomized first
        :param ig: inputgen
        """

        self.ids=ids
        self.ig=ig

    def __getitem__(self, index):
        return self.ig.get_by_id(self.ids[index])

    def __len__(self):
        return len(self.ids)


class InputGenE(InputGen):
    def __init__(self, death_fold=5, validation_proportion=0.1, random_seed=12345):
        verbose=False
        debug=True
        super(InputGenE, self).__init__(verbose=verbose, debug=debug)
        self.death_fold=death_fold
        self.validation_proportion=validation_proportion
        self.random_seed=random_seed
        np.random.seed(random_seed)

        self.valid=None


    def train_valid_split(self):
        # splits the whole set by id
        death_rep_person_id = self.death.index.get_level_values(0).unique().values
        death_rep_person_id=permutation(death_rep_person_id)
        valid_death_len=int(len(death_rep_person_id)*self.validation_proportion)
        self.valid_death_id=death_rep_person_id[:valid_death_len]
        self.train_death_id=death_rep_person_id[valid_death_len:]

        no_death_rep_person_id = self.rep_person_id[np.invert(np.in1d(self.rep_person_id, death_rep_person_id))]
        no_death_rep_person_id=permutation(no_death_rep_person_id)
        valid_no_death_len=int(len(no_death_rep_person_id)*self.validation_proportion)
        self.valid_no_death_id=no_death_rep_person_id[:valid_no_death_len]
        self.train_no_death_id=no_death_rep_person_id[valid_no_death_len:]

    def get_valid_dataset(self):
        """
        should be run only once in its lifetime
        :return:
        """
        if self.valid is None:
            ids=np.concatenate((self.valid_death_id,self.valid_no_death_id))
            ids=permutation(ids)
            self.valid=GenHelper(ids, self)
        return self.valid

    def get_train_dataset(self):
        """
        modifies the deathfold everytime it is called
        :return:
        """
        resample_rate=2**self.death_fold
        new_no_death_length=len(self.train_no_death_id)//resample_rate
        new_no_death_id=permutation(self.train_no_death_id)
        new_no_death_id=new_no_death_id[:new_no_death_length]
        ids=np.concatenate((self.train_death_id,new_no_death_id))
        ids=permutation(ids)
        train=GenHelper(ids,self)
        return train

    def change_fold(self):
        if self.death_fold==0:
            print("death fold is zero")
        else:
            self.death_fold-=1

    def __getitem__(self, item):
        raise NotImplementedError("Do not call this function")



if __name__=="__main__":
    ig=InputGenE()
    ig.train_valid_split()
    valid=ig.get_valid_dataset()
    train=ig.get_train_dataset()
    print(valid[100])
    print(train[192])
    print("Done")