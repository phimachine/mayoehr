from death.post.dfmanager import DFManager
from torch.utils.data import Dataset, DataLoader
import numpy as np

# we can only assume that all deaths are recorded

# https://github.com/utkuozbulak/pytorch-custom-dataset-examples

class InputGen(Dataset):
    '''
    take a data frame manager object and produce inputs wrapped in torch objects
    '''
    def __init__(self,load_pickle=True,verbose=False):
        self.dfs=DFManager()
        self.rep_id=self.dfs.demo["rep_person_id"]

        self.dfs.load_pickle(verbose=verbose)
        print('reached this step')


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

        ### we pull all relevant data
        id=self.rep_id[index]

        ### we compile it into time series


    def __len__(self):
        '''
        Length of the demographics dataset
        :return:
        '''

if __name__=="__main__":
    ig=InputGen(load_pickle=True)
