from death.post.qdata import Dfs
from torch.utils.data import Dataset, DataLoader

# TODO coerce that there is no empty inputs, we can only assume that all deaths are recorded

# https://github.com/utkuozbulak/pytorch-custom-dataset-examples

class InputGen(Dataset):
    '''
    take a dataframe manager object and produce inputs wrapped in torch objects
    '''
    def __init__(self):
        self.dfs=Dfs()

    def __getitem__(self, index):
        '''
        pulls a row in demographics
        pulls all data from all files
        compile it into a longest vector
        construct label from death records

        We do not use sliding window augmentation here.
        We do sliding window augmentation probably in the training stage.
        Simply because we cannot find the index for a window without loading the series.

        :param index:
        :return:
        '''

        pass

    def

    def __len__(self):
        '''
        Length of the demographics dataset
        :return:
        '''