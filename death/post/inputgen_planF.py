# Plan E does the train valid test split. Because we are doing model selection very aggressively,
# this should prevent validation leakage.
# The primary goal of Plan E is to do the curriculum learning. The reason is because our model overfitting is
# very significant. Basic knowledge about overfitting, the reason for overfitting are these three:
# 1, bad model choice
# 2, over-parameterization
# 3, lack of data
# 4, distribution mismatch.

# Reason 2 and 3 are well known. This is the typical VC dimension trade-off
# With 1) bad model choice, the prior distribution fundamentally mismatches the target distribution,
# and optimization does not help that, because prior blocks the learning of useful features.
# For example, a model that a priori chose not to model visual channel will not be able to learn
# recognize objects by pictures. The model will be forced to learn the noise distribution, causing overfitting.
# This is a problem we cannot solve, unfortunately. If DNC for some reason is fundamentally worse than LSTM,
# then we learnt our lesson and conclude our research. DNC was a promising candidate, and we wouldn't know if it's
# bad before we try it.

# With 4), the training (and learnt) distribution is fundamentally different from the target distribution.
# The model can be very good at predicting the training distribution, but if at validation/test time,
# another distribution was sampled to test the model, the model performance will be bottle-necked.
# The model will continue to learn the details of the training distribution, and training loss will go down,
# but the lessons might not be homogeneously applied to the validation time.

# Before we step into curriculum learning, we should probably try to train it with zero death_fold.
# 4 months ago I tried to train with zero death_fold and the training got stuck because of the sparsity.
# 4 months have passed and the model has changed a lot. Now I'm not so sure if death_fold was the problem initially.
# Let's try with zero death_fold to ascertain that. See if training converges to trivial.
# I think it should. Initialization of neural network is very important. Bad initialization with difficult
# training set causes the gradient descent to converge to zero.


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


class InputGenF(InputGen):
    def __init__(self, death_fold=5, curriculum=False, validation_test_proportion=0.1, random_seed=54321):
        verbose=False
        debug=True
        super(InputGenF, self).__init__(verbose=verbose, debug=debug)
        self.death_fold=death_fold
        self.validation_test_proportion=validation_test_proportion
        self.random_seed=random_seed
        np.random.seed(random_seed)
        # for curriculum learning, every time the training set is requested, the death proportion will be adjusted
        self.curriculum=curriculum

        self.valid=None
        self.test=None
        self.train_valid_test_split()

        # the proportion of death records the last training set has
        self.proportion=None
        print("Using InputGenF")
        if self.curriculum:
            print("Using curriculum learning")
        else:
            print("Not using curriculum learning")


    def train_valid_test_split(self):
        # splits the whole set by id
        death_rep_person_id = self.death.index.get_level_values(0).unique().values
        death_rep_person_id=permutation(death_rep_person_id)
        valid_or_test_death_len=int(len(death_rep_person_id)*self.validation_test_proportion)
        self.valid_death_id=death_rep_person_id[:valid_or_test_death_len]
        self.test_death_id=death_rep_person_id[valid_or_test_death_len:valid_or_test_death_len*2]
        self.train_death_id=death_rep_person_id[valid_or_test_death_len*2:]

        no_death_rep_person_id = self.rep_person_id[np.invert(np.in1d(self.rep_person_id, death_rep_person_id))]
        no_death_rep_person_id=permutation(no_death_rep_person_id)
        valid_or_test_no_death_len=int(len(no_death_rep_person_id)*self.validation_test_proportion)
        self.valid_no_death_id=no_death_rep_person_id[:valid_or_test_no_death_len]
        self.test_no_death_id=death_rep_person_id[valid_or_test_no_death_len:valid_or_test_no_death_len*2]
        self.train_no_death_id=no_death_rep_person_id[valid_or_test_no_death_len*2:]

    def get_valid(self):
        """
        should be run only once in its lifetime
        :return:
        """
        if self.valid is None:
            ids=np.concatenate((self.valid_death_id,self.valid_no_death_id))
            ids=permutation(ids)
            self.valid=GenHelper(ids, self)
        return self.valid

    def get_test(self):
        """
        should be run only once in its lifetime
        :return:
        """
        if self.test is None:
            ids=np.concatenate((self.test_death_id,self.test_no_death_id))
            ids=permutation(ids)
            self.test=GenHelper(ids, self)
        return self.test

    def get_train(self):
        """
        modifies the deathfold everytime it is called
        :return:
        """
        resample_rate=2**self.death_fold
        new_no_death_length=len(self.train_no_death_id)//resample_rate
        new_no_death_id=permutation(self.train_no_death_id)
        new_no_death_id=new_no_death_id[:new_no_death_length]
        self.proportion=len(self.train_death_id)/(len(self.train_death_id)+len(new_no_death_id))
        print("Death proportion", self.proportion, ", death fold",  self.death_fold)
        ids=np.concatenate((self.train_death_id,new_no_death_id))
        ids=permutation(ids)
        train=GenHelper(ids,self)
        if self.curriculum:
            self.change_fold()
        return train

    def change_fold(self):
        if self.death_fold==0:
            print("Death fold is zero, death fold not changed")
        else:
            self.death_fold-=1

    def __getitem__(self, item):
        raise NotImplementedError("Do not call this function")



if __name__=="__main__":
    ig=InputGenF()
    ig.train_valid_test_split()
    valid=ig.get_valid()
    train=ig.get_train()
    print(valid[100])
    print(train[192])
    print("Done")