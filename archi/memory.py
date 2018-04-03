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
        memory_usage_u_t=torch.Tensor(param.N).zero_()

    def content_lookup(self, desired_content, memory, key_strength):
        '''

        :param desired_content: \k, (W), R, lookup key
        :param memory: M, (param.N, param.W)
        :param key_strength: \beta, (1) [1, \infty)
        :param index: i, lookup on memory[i]
        :return: most weighted similar: C(M,k,\beta), (param.N, 1), (0,1)
        '''

        # TODO make sure the dimensions are correct.
        similarties= cosine_similarity(desired_content, memory)
        weighted=similarties*key_strength
        return softmax(weighted)


    # the highest freed will be retained? What does it mean?
    def memory_retention(self,free_gate, read_weighting):
        '''

        :param free_gate: f, (R), [0,1], from interface vector
        :param read_weighting: TODO calculated later, w, (N, R), simplex bounded,
               note it's from previous timestep.
        :return: \psi, (N), simplex bounded
        '''


        # a free gate belongs to a read head.
        # a single read head weighting is a (N) dimensional simplex bounded value

        # (N, R) TODO make sure this is pointwise multiplication, not matmul
        inside_bracket = 1 - read_weighting * free_gate
        return torch.prod(inside_bracket, 1)

    def usage_vector(self, previous_usage, write_wighting, memory_retention):
        '''
        I cannot understand what this vector is for.
        This should be a variable that records the usages history of a vector
        Penalized by the free gate.

        :param previous_usage: u_{t-1}, (N), [0,1]
        :param write_wighting: w^w_{t-1}, (N), (inferred) sum to one
        :param memory_retention: \psi_t, (N), simplex bounded
        :return: u_t, (N), [0,1], the next usage,
        '''

        ret= (previous_usage+write_wighting-previous_usage*write_wighting)*memory_retention

        return ret

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

        # the indices here is \phi_t referred to in the paper
        sorted, indices=usage_vector.sort()
        ret=torch.Tensor(param.N)
        rolling_product=1
        ret[indices[0]]=(1-sorted[0])
        print("retret:", ret)
        print(sorted)
        for i in range(param.N-1):
            rolling_product=sorted[i]*rolling_product
            print("rolling product: ", rolling_product)
            ret[indices[i+1]]=(1-sorted[i])*rolling_product
        print("ret: ",ret)
