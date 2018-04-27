# parameter notation convention:
# latex symbols
# :param variable_name: variable_symbol, tensor dimension, domain, note

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, softmax, normalize
import archi.param as param
from torch.autograd import Variable

class Memory(nn.Module):

    def __init__(self):
        super(Memory, self).__init__()
        # u_0
        self.usage_vector=torch.Tensor(param.N).zero_()
        # p, (W), should be simplex bound
        self.precedence_weighting=torch.Tensor(param.N).zero_()
        # (N,N)
        self.temporal_memory_linkage=torch.Tensor(param.N, param.N).zero_()
        #TODO will autograd alter memory? Should autograd alter memory?
        # (N,W)
        self.memory=Variable(torch.Tensor(param.N,param.W),requires_grad=False)
        # (N, R). Does this require gradient?
        self.read_weightings=torch.Tensor(param.N,param.R)

    def reset_parameters(self):
        self.usage_vector.zero_()
        self.precedence_weighting.zero_()
        self.temporal_memory_linkage.zero_()
        # I did not find reference here:
        self.memory.zero_()
        self.read_weightings.fill_(1)

    def write_content_weighting(self, write_key, key_strength):
        '''

        :param memory: M, (N, W)
        :param write_key: k, (W), R, desired content
        :param key_strength: \beta, (1) [1, \infty)
        :param index: i, lookup on memory[i]
        :return: most similar weighted: C(M,k,\beta), (N, 1), (0,1)
        '''

        # TODO make sure the dimensions are correct.
        similarties= cosine_similarity(self.memory,write_key,dim=-1)
        weighted=similarties*key_strength
        return softmax(weighted,dim=0)

    def read_content_weighting(self, read_keys, key_strengths):
        '''

        :param memory: M, (N, W)
        :param read_keys: k^r_t, (W,R), R, desired content
        :param key_strength: \beta, (1) [1, \infty)
        :param index: i, lookup on memory[i]
        :return: most similar weighted: C(M,k,\beta), (N, 1), (0,1)
        '''

        # TODO make sure the dimensions are correct.
        '''
            torch definition
            def cosine_similarity(x1, x2, dim=1, eps=1e-8):
                w12 = torch.sum(x1 * x2, dim)
                w1 = torch.norm(x1, 2, dim)
                w2 = torch.norm(x2, 2, dim)
                return w12 / (w1 * w2).clamp(min=eps)
        '''

        outer=torch.matmul(self.memory,read_keys)
        # print(outer)

        # this is confusing. matrix[n] access nth row, not column
        # this is very counter-intuitive, since columns have meaning,
        # because they represent vectors
        mem_norm=torch.norm(self.memory,p=2,dim=1)
        # print("mem_norm:\n",mem_norm)
        read_norm=torch.norm(read_keys,p=2,dim=0)
        # print(read_norm)
        mem_norm=mem_norm.unsqueeze(1)
        read_norm=read_norm.unsqueeze(0)
        normalizer=torch.matmul(mem_norm,read_norm)
        # print("normalizer:\n",normalizer)

        # if transposed then similiarities[0] refers to the first read key
        similarties= outer/normalizer
        weighted=similarties*key_strengths
        # print("weighted:\n",weighted)
        return softmax(weighted,dim=1)

    # the highest freed will be retained? What does it mean?
    def memory_retention(self,free_gate):
        '''

        :param free_gate: f, (R), [0,1], from interface vector
        :param read_weighting: w^r_t, (N, R), simplex bounded,
               note it's from previous timestep.
        :return: \psi, (N), simplex bounded
        '''

        # a free gate belongs to a read head.
        # a single read head weighting is a (N) dimensional simplex bounded value

        # (N, R) TODO make sure this is pointwise multiplication, not matmul
        inside_bracket = 1 - self.read_weightings * free_gate
        return torch.prod(inside_bracket, 1)


    def update_usage_vector(self, write_wighting, memory_retention):
        '''
        TODO need to review the meaning of this vector

        :param previous_usage: u_{t-1}, (N), [0,1]
        :param write_wighting: w^w_{t-1}, (N), (inferred) sum to one
        :param memory_retention: \psi_t, (N), simplex bounded
        :return: u_t, (N), [0,1], the next usage,
        '''

        ret= (self.usage_vector+write_wighting-self.usage_vector*write_wighting)*memory_retention
        self.usage_vector=ret
        return ret


    def allocation_weighting(self,usage_vector):
        '''
        Sorts the memory by usages first.
        Then perform calculation depending on the sort order.

        The alloation_weighting of the third least used memory is calculated as follows:
        Find the least used and second least used. Multiply their usages.
        Multiply the product with (1-usage of the third least), return.

        Do not confuse the sort order and the memory's natural location.
        Verify backprop.

        :param usage_vector: u_t, (N), [0,1]
        :return:
        '''

        sorted, indices= usage_vector.sort()
        cum_prod=torch.cumprod(sorted,0)
        # notice the index on the product
        # TODO this does not deal with batch inputs
        cum_prod=torch.cat([torch.Tensor([1]),cum_prod],0)[:-1]
        sorted_inv=1-sorted
        allocation_weighting=sorted_inv*cum_prod
        # to shuffle back in place
        return allocation_weighting.index_select(0, indices)


    def write_weighting(self, write_key, write_strength, allocation_gate, write_gate, allocation_weighting):
        '''
        calculates the weighting on each memory cell when writing a new value in

        :param memory: M, (N, W), memory block
        :param write_key: k^w_t, (W), R, the key that is to be written
        :param write_strength: \beta, (1) bigger it is, stronger it concentrates the content weighting
        :param allocation_gate: g^a_t, (1), balances between write by content and write by allocation gate
        :param write_gate: g^w_t, (1), overall strength of the write signal
        :param allocation_weighting: see above.
        :return: write_weighting: (N), simplex bound
        '''

        # measures content similarity
        content_weighting=self.write_content_weighting(write_key,write_strength)
        write_weighting=write_gate*(allocation_gate*allocation_weighting+(1-allocation_gate)*content_weighting)
        return write_weighting

    def update_precedence_weighting(self,write_weighting):
        '''

        :param write_weighting: (N)
        :return: self.precedence_weighting: (N), simplex bound
        '''
        sum_ww=sum(write_weighting)
        self.precedence_weighting=(1-sum_ww)*self.precedence_weighting+write_weighting
        return self.precedence_weighting

    def update_temporal_linkage_matrix(self,write_weighting):
        '''

        :param write_weighting: (N)
        :param precedence_weighting: (N), simplex bound
        :return:
        '''

        ww_j=write_weighting.unsqueeze(0).expand(param.N,-1)
        ww_i=write_weighting.unsqueeze(1).expand(-1,param.N)
        p_j=self.precedence_weighting.unsqueeze(0).expand(param.N,-1)

        self.temporal_memory_linkage= (1 - ww_j - ww_i) * self.temporal_memory_linkage + ww_i * p_j
        return self.temporal_memory_linkage

    def backward_weighting(self):
        '''

        :return: backward_weighting: b^i_t, (N,R)
        '''
        return torch.matmul(self.temporal_memory_linkage,self.read_weightings)

    def forward_weighting(self):
        '''

        :return: forward_weighting: f^i_t, (N,R)
        '''
        return torch.matmul(self.temporal_memory_linkage.t(),self.read_weightings)

    # TODO sparse update, skipped because it's for performance improvement.

    def read_weighting(self,forward_weighting, backward_weighting, read_keys,
                         read_key_strengths, read_mode_weighting):
        '''

        :param forward_weighting: (N,R)
        :param backward_weighting: (N,R)
        :param read_keys: k^w_t, (W,R)
        :param read_key_strengths:
        :param read_mode_weighting: /pi_t^i, (R,3)
        :return:
        '''

        content_weighting=self.read_content_weighting(read_keys,read_key_strengths)
        read_weighting_i=read_mode_weighting[0]*backward_weighting+\
                         read_mode_weighting[1]*content_weighting+\
                         read_mode_weighting[2]*forward_weighting

        return read_weighting_i

    def read_memory(self):
        # this is currently the formula of a single read head TODO
        '''

        :return: read_vectors: r^i_t
        '''
        return self.memory.t()*self.read_weightings

    def write_to_memory(self,write_weighting,erase_vector,write_vector):
        '''

        :param write_weighting: the strength of writing
        :param erase_vector:
        :param write_vector: what to write, a cat picture, e.g.
        :return:
        '''

        self.memory=self.memory*(torch.ones((param.N,param.W))-write_weighting*
                                 erase_vector.t())+write_weighting*write_vector.t()

    def forward(self,read_keys,read_key_strengths,read_mode_vectors,write_key,write_strength,allocation_gate,
                write_gate,erase_vector,write_vector,free_gate):
        # read from memory first
        #TODO list is not okay, and we are going to separate read and write weighting functions
        self.rwis=[]
        for read_key, read_key_strength, read_mode_vector in zip(read_keys,read_key_strengths,read_mode_vectors):
            self.rwis.append(self.read_weighting_i(self.forward_weighting(), self.backward_weighting()),
                            read_key,read_key_strength, read_mode_vector)
        read_weighting=torch.Tensor(self.rwis)
        ri=[]
        for rwi in self.rwis:
            ri.append(self.read_memory_i(read_weighting_i=rwi))
        # write to memory
        allocation_weighting=self.allocation_weighting(self.usage_vector)
        write_weighting=self.write_weighting(write_key,write_strength,allocation_gate,write_gate,allocation_weighting)
        self.write_to_memory(write_weighting=write_weighting,erase_vector=erase_vector,write_vector=write_vector)

        # update memory
        memory_retention=self.memory_retention(free_gate,read_weighting)
        self.update_usage_vector(write_weighting,memory_retention)
        self.update_temporal_linkage_matrix(write_weighting,self.precedence_weighting)
        self.update_precedence_weighting(write_weighting)
