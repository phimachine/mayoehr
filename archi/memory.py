# this is the memory module
# wow the dynamic memory thing is hard.
# cannot read the paper anymore. let's just implement it. hopefully by the end
# I know what I'm doing.

# parameter notation convention:
# latex symbols
# :param variable_name: variable_symbol, tensor dimension, domain, note

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, softmax
import archi.param as param

class Memory(nn.Module):

    def __init__(self):
        self.memory_usage=torch.Tensor(param.N).zero_()
        # p, (W), should be simplex bound
        self.precedence_weighting=torch.Tensor(param.N).zero_()
        self.temporal_memory_linkage=torch.Tensor(param.N,param.N).zero_()

    def content_weighting(self, memory, write_key, key_strength):
        '''

        :param memory: M, (N, W)
        :param write_key: k, (W), R, desired content
        :param key_strength: \beta, (1) [1, \infty)
        :param index: i, lookup on memory[i]
        :return: most similar weighted: C(M,k,\beta), (N, 1), (0,1)
        '''

        # TODO make sure the dimensions are correct.
        similarties= cosine_similarity(write_key, memory)
        weighted=similarties*key_strength
        return softmax(weighted)


    # the highest freed will be retained? What does it mean?
    def memory_retention(self,free_gate, read_weighting):
        '''

        :param free_gate: f, (R), [0,1], from interface vector
        :param read_weighting: TODO calculated later, w, (N, R), simplex bounded,
               note it's from previous timestep.
        :return: \phi, (N), simplex bounded
        '''


        # a free gate belongs to a read head.
        # a single read head weighting is a (N) dimensional simplex bounded value

        # (N, R) TODO make sure this is pointwise multiplication, not matmul
        inside_bracket = 1 - read_weighting * free_gate
        return torch.prod(inside_bracket, 1)

    #cumprod!

    def usage_vector(self, previous_usage, write_wighting, memory_retention):
        '''
        I cannot understand what this vector is for.
        This should be a variable that records the usages history of a vector
        Penalized by the free gate.

        :param previous_usage: u_{t-1}, (N), [0,1]
        :param write_wighting: w^w_{t-1}, (N), (inferred) sum to one
        :param memory_retention: \phi_t, (N), simplex bounded
        :return: u_t, (N), [0,1], the next usage
        '''

        ret= (previous_usage+write_wighting-previous_usage*write_wighting)*memory_retention
        self.memory_usage=ret

    def allocation_weighting(self,usage_vector):
        '''
        Sorts the memory by usages first.
        Then perform calculation depending on the sort order.

        The alloation_weighting of the third least used memory is calculated as follows:
        Find the least used and second least used. Multiply their usages.
        Multiply the product with (1-usage of the third least), return.

        TODO
        Do not confuse the sort order and the memory's natural location.
        Verify backprop.

        :param usage_vector: u_t, (N), [0,1]
        :return:
        '''

        sorted, indices= usage_vector.sort()


    def write_weighting(self,memory, write_key, write_strength, allocation_gate, write_gate, allocation_weighting):

        '''
        calculates the weighting on each memory cell when writing a new value in

        :param memory: M, (N, W), memory block
        :param write_key: k^w_t, (W), R, the key that is to be written
        :param write_strength: \beta, (1) bigger it is, stronger it concentrates the content weighting
        :param allocation_gate: g^a_t, (1), balances between write by content and write by allocation gate
        :param write_gate: g^w_t, overall strength of the write signal
        :param allocation_weighting: see above.
        :return: write_weighting: where does this write key go?
        '''

        # measures content similarity
        content_weighting=self.content_weighting(memory,write_key,write_strength)
        write_weighting=write_gate*(allocation_gate*allocation_weighting+(1-allocation_gate)*content_weighting)
        return write_weighting

    def update_temporal_memory_linkage(self,write_weighting):
        sum_ww=sum(write_weighting)
        self.precedence_weighting=(1-sum_ww)*self.precedence_weighting+write_weighting
        
