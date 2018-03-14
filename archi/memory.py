# this is the memory module
# wow the dynamic memory thing is hard.
# cannot read the paper anymore. let's just implement it. hopefully by the end
# I know what I'm doing.

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, softmax
import archi.param as param

class Memory(nn.Module):

    def __init__(self):
        memory_usage_u_t=torch.Tensor(param.N).zero_()

    def content_lookup(self, desired_content, memory, key_strength):
        '''

        :param desired_content: \k, (W), R, lookup key
        :param memory: M, (param.N, param.W)
        :param key_strength: \beta, (1) [1, \infty)
        :param index: i, lookup on memory[i]
        :return: most weighted similar: C, (param.N, 1), (0,1)
        '''

        # TODO make sure the dimensions are correct.
        similarties= cosine_similarity(desired_content, memory)
        weighted=similarties*key_strength
        return softmax(weighted)


    # the highest freed will be retained? What does it mean?
    def memory_retention(self,free_gate, read_weighting):
        '''

        :param free_gate: f, (R), [0,1], from interface vector
        :param read_weighting: TODO calculated later, w, (N, R), simplex bounded
        :return: \phi, (N), simplex bounded
        '''


        # a free gate belongs to a read head.
        # a single read head weighting is a (N) dimensional simplex bounded value

        # (N, R) TODO make sure this is pointwise multiplication, not matmul
        inside_bracket = 1 - read_weighting * free_gate
        return torch.prod(inside_bracket, 1)




