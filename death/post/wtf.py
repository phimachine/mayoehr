from death.post.qdata import DFManager
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



    def __len__(self):
        '''
        Length of the demographics dataset
        :return:
        '''

if __name__=="__main__":
    ig=InputGen(load_pickle=True)
