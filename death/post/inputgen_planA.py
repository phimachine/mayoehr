from death.post.qdata import Dfs
from torch.utils.data import Dataset, DataLoader
import numpy as np

# we can only assume that all deaths are recorded

# https://github.com/utkuozbulak/pytorch-custom-dataset-examples

class InputGen(Dataset):
    '''
    take a dataframe manager object and produce inputs wrapped in torch objects
    '''
    def __init__(self):
        self.dfs=Dfs()
        self.rep_id=self.dfs.demo["rep_person_id"]

    def __getitem__(self, index):
        '''
        pulls a row in demographics
        pulls all data from all files
        compile it into a longest vector
        construct label from death records

        We do not use sliding window augmentation here.
        We do sliding window augmentation probably in the training stage.
        Simply because we cannot find the index for a window without loading the series.

        input will be variable length, with a unit of a month

        :param index:
        :return: (time, longest)
        '''

        id=self.rep_id[index]
        # find death targets
        


    def __len__(self):
        '''
        Length of the demographics dataset
        :return:
        '''

if __name__=="__main__":
    ig=InputGen()