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
        self.usage_vector=torch.Tensor(param.N)
        # p, (N), should be simplex bound
        self.precedence_weighting=torch.Tensor(param.N)
        # (N,N)
        self.temporal_memory_linkage=torch.Tensor(param.N, param.N)
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

    def new_sequence_reset(self):
        # study this function
        # memory is the only value that is not reset after new sequence
        self.temporal_memory_linkage.zero_()
        self.precedence_weighting.zero_()
        self.usage_vector.zero_()

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
        :return: most similar weighted: C(M,k,\beta), (N, R), (0,1)
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
        :param write_wighting: w^w_{t-1}, (N), simplex bound
        :param memory_retention: \psi_t, (N), simplex bound
        :return: u_t, (N), [0,1], the next usage,
        '''

        ret= (self.usage_vector+write_wighting-self.usage_vector*write_wighting)*memory_retention
        self.usage_vector=ret
        return ret


    def allocation_weighting(self):
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

        # this should not be an in place sort.
        sorted, indices= self.usage_vector.sort()
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
        :return: updated_temporal_linkage_matrix
        '''

        ww_j=write_weighting.unsqueeze(1).expand(-1,param.N,-1)
        ww_i=write_weighting.unsqueeze(2).expand(-1,-1,param.N)
        p_j=self.precedence_weighting.unsqueeze(0).expand(param.N,-1)
        batch_temporal_memory_linkage=self.temporal_memory_linkage.expand(param.bs,-1,-1)

        self.temporal_memory_linkage= (1 - ww_j - ww_i) * batch_temporal_memory_linkage + ww_i * p_j
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

    def update_read_weightings(self,forward_weighting, backward_weighting, read_keys,
                         read_strengths, read_modes):
        '''

        :param forward_weighting: (N,R)
        :param backward_weighting: (N,R)
        ****** read_content_weighting: C, (N,R), (0,1)
        :param read_keys: k^w_t, (W,R)
        :param read_key_strengths: (R)
        :param read_modes: /pi_t^i, (R,3)
        :return: read_weightings: w^r_t, (N,R)
        '''

        content_weighting=self.read_content_weighting(read_keys,read_strengths)
        # has dimension (3,N,R)
        all_weightings=torch.stack([backward_weighting,content_weighting,forward_weighting])
        # permute to dimension (R,N,3)
        all_weightings=all_weightings.permute(2,1,0)
        # this is becuase torch.matmul is designed to iterate all dimension excluding the last two
        # dimension (R,3,1)
        read_modes=read_modes.unsqueeze(2)
        # dimension (N,R)
        self.read_weightings=torch.matmul(all_weightings,read_modes).squeeze(2).t()
        return self.read_weightings

    def read_memory(self):
        '''

        memory: (N,W)
        read weightings: (N,R)

        :return: read_vectors: [r^i_R], (W,R)
        '''

        return torch.matmul(self.memory.t(),self.read_weightings)

    def write_to_memory(self,write_weighting,erase_vector,write_vector):
        '''

        :param write_weighting: the strength of writing
        :param erase_vector: e_t, (W), [0,1]
        :param write_vector: w^w_t, (W), R
        :return:
        '''

        term1_2=torch.ger(write_weighting,erase_vector)
        term1=self.memory*(torch.ones((param.N,param.W))-term1_2)
        term2=torch.ger(write_weighting,write_vector)

        self.memory=term1+term2


    def forward(self,read_keys, read_strengths, write_key, write_strength,
                erase_vector, write_vector, free_gates, allocation_gate,
                write_gate, read_modes):

        # return read_vectors: [r^i_R], (W,R)


        # read from memory first
        read_vectors=self.read_memory()
        # then write
        allocation_weighting=self.allocation_weighting()
        write_weighting=self.write_weighting(write_key,write_strength,
                                             allocation_gate,write_gate,allocation_weighting)
        self.write_to_memory(write_weighting,erase_vector,write_vector)
        # update some
        memory_retention = self.memory_retention(free_gates)
        self.update_usage_vector(write_weighting, memory_retention)
        self.update_temporal_linkage_matrix(write_weighting)
        self.update_precedence_weighting(write_weighting)

        forward_weighting=self.forward_weighting()
        backward_weighting=self.backward_weighting()

        self.update_read_weightings(forward_weighting,backward_weighting,read_keys,read_strengths,
                    read_modes)

        return read_vectors


        # OLD CODE:
        #
        # #TODOO list is not okay, and we are going to separate read and write weighting functions
        # read_weightings=self.read_weightings
        # self.rwis=[]
        # for read_key, read_key_strength, read_mode_vector in zip(read_keys,read_strengths,read_modes):
        #     self.rwis.append(self.read_weighting_i(self.forward_weighting(), self.backward_weighting()),
        #                     read_key,read_key_strength, read_modes)
        # read_weighting=torch.Tensor(self.rwis)
        # ri=[]
        # for rwi in self.rwis:
        #     ri.append(self.read_memory_i(read_weighting_i=rwi))
        #
        # # write to memory
        # allocation_weighting=self.allocation_weighting(self.usage_vector)
        # write_weighting=self.write_weighting(write_key,write_strength,allocation_gate,write_gate,allocation_weighting)
        # self.write_to_memory(write_weighting=write_weighting,erase_vector=erase_vector,write_vector=write_vector)
        #
        # # update memory
        # memory_retention=self.memory_retention(free_gate,read_weighting)
        # self.update_usage_vector(write_weighting,memory_retention)
        # self.update_temporal_linkage_matrix(write_weighting,self.precedence_weighting)
        # self.update_precedence_weighting(write_weighting)
