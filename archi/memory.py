# this is the memory module
# wow the dynamic memory thing is HARD.

import torch.nn as nn
from torch.nn.functional import cosine_similarity

class Memory(nn.Module):

    def __init__(self):
        pass

    def content_lookup(self,lookup_key, memory, key_strength, index):
        '''

        :param lookup_key: (W) \in R
        :param memory: (param_N, param_W)
        :param key_strength: (1) \in [1, \infty)
        :param index: lookup on memory[i]
        :return:
        '''

