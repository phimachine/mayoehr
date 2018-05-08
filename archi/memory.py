# parameter notation convention:
# latex symbols
# :param variable_name: variable_symbol, tensor dimension, domain, note

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, softmax, normalize
import archi.param as param
from torch.autograd import Variable
import pdb
import numpy

class Memory(nn.Module):

    def __init__(self):
        super(Memory, self).__init__()
        # u_0
        self.usage_vector=torch.Tensor(param.bs,param.N)
        # p, (N), should be simplex bound
        self.precedence_weighting=torch.Tensor(param.bs,param.N)
        # (N,N)
        self.temporal_memory_linkage=torch.Tensor(param.bs,param.N, param.N)
        # (N,W)
        self.memory=torch.Tensor(param.N,param.W)
        self.previous_memory=None
        # (N, R). Does this require gradient?
        self.last_read_weightings=torch.Tensor(param.bs, param.N, param.R)

    def reset_parameters(self):
        self.usage_vector.zero_()
        self.precedence_weighting.zero_()
        self.temporal_memory_linkage.zero_()
        self.memory.zero_()
        self.memory.requires_grad_()
        self.last_read_weightings.fill_(1.0/param.N)

    def new_sequence_reset(self):
        # study this function
        # memory is the only value that is not reset after new sequence
        self.temporal_memory_linkage.zero_()
        self.precedence_weighting.zero_()
        self.usage_vector.zero_()

    def write_content_weighting(self, write_key, key_strength, eps=1e-8):
        '''

        :param memory: M, (N, W)
        :param write_key: k, (W), R, desired content
        :param key_strength: \beta, (1) [1, \infty)
        :param index: i, lookup on memory[i]
        :return: most similar weighted: C(M,k,\beta), (N), (0,1)
        '''

        # memory will be (N,W)
        # write_key will be (bs, W)
        # I expect a return of (N,bs), which marks the similiarity of each W with each mem loc

        # (param.bs, param.N)
        innerprod=torch.matmul(write_key,self.memory.t())
        # (parm.N)
        memnorm=torch.norm(self.memory,2,1)
        # (param.bs)
        writenorm=torch.norm(write_key,2,1)
        # (param.N, param.bs)
        normalizer=torch.ger(memnorm,writenorm)
        similarties=innerprod/normalizer.t().clamp(min=eps)
        similarties=similarties*key_strength.expand(-1,param.N)
        normalized= softmax(similarties,dim=1)
        return normalized

    def read_content_weighting(self, read_keys, key_strengths, eps=1e-8):
        '''

        :param memory: M, (N, W)
        :param read_keys: k^r_t, (W,R), R, desired content
        :param key_strength: \beta, (R) [1, \infty)
        :param index: i, lookup on memory[i]
        :return: most similar weighted: C(M,k,\beta), (N, R), (0,1)
        '''

        '''
            torch definition
            def cosine_similarity(x1, x2, dim=1, eps=1e-8):
                w12 = torch.sum(x1 * x2, dim)
                w1 = torch.norm(x1, 2, dim)
                w2 = torch.norm(x2, 2, dim)
                return w12 / (w1 * w2).clamp(min=eps)
        '''

        innerprod=torch.matmul(self.memory.unsqueeze(0),read_keys)

        # this is confusing. matrix[n] access nth row, not column
        # this is very counter-intuitive, since columns have meaning,
        # because they represent vectors
        mem_norm=torch.norm(self.memory,p=2,dim=1)
        read_norm=torch.norm(read_keys,p=2,dim=1)
        mem_norm=mem_norm.unsqueeze(1)
        read_norm=read_norm.unsqueeze(1)
        # (batch_size, locations, read_heads)
        normalizer=torch.matmul(mem_norm,read_norm)

        # if transposed then similiarities[0] refers to the first read key
        similarties= innerprod/normalizer.clamp(min=eps)
        weighted=similarties*key_strengths.unsqueeze(1).expand(-1,param.N,-1)
        ret= softmax(weighted,dim=1)
        return ret

    # the highest freed will be retained? What does it mean?
    def memory_retention(self,free_gate):
        '''

        :param free_gate: f, (R), [0,1], from interface vector
        :param read_weighting: w^r_t, (N, R), simplex bounded,
               note it's from previous timestep.
        :return: \psi, (N), [0,1]
        '''

        # a free gate belongs to a read head.
        # a single read head weighting is a (N) dimensional simplex bounded value

        # (N, R) TODO make sure this is pointwise multiplication, not matmul
        inside_bracket = 1 - self.last_read_weightings * free_gate.unsqueeze(1).expand(-1,param.N,-1)
        ret= torch.prod(inside_bracket, 2)
        if (ret<0).any() or (ret>1).any():
            raise ValueError("memory retention exceeded limit")
        return ret

    def update_usage_vector(self, write_wighting, memory_retention):
        '''

        :param previous_usage: u_{t-1}, (N), [0,1]
        :param write_wighting: w^w_{t-1}, (N), simplex bound
        :param memory_retention: \psi_t, (N), simplex bound
        :return: u_t, (N), [0,1], the next usage,
        '''

        ret= (self.usage_vector+write_wighting-self.usage_vector*write_wighting)*memory_retention
        if (ret>1).any() or (ret<0).any():
            raise ValueError("A usage vector exceeded the bound")
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
        :return: allocation_wighting: a_t, (N), simplex bound
        '''
        # TODO This function has a bug
        # this should not be an in place sort.
        sorted, indices= self.usage_vector.sort(dim=1)
        cum_prod=torch.cumprod(sorted,1)
        # notice the index on the product
        cum_prod=torch.cat([torch.ones(param.bs,1),cum_prod],1)[:,:-1]
        sorted_inv=1-sorted
        allocation_weighting=sorted_inv*cum_prod
        # to shuffle back in place
        ret=torch.gather(allocation_weighting,1,indices)
        if (ret.sum(1)>1).any() or (ret<0).any():
            raise ValueError("allocation weighting simplex bound problem.")
        return ret
        # return allocation_weighting.index_select(0, indices)


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
        # TODO this function has a bug
        # measures content similarity
        content_weighting=self.write_content_weighting(write_key,write_strength)
        write_weighting=write_gate*(allocation_gate*allocation_weighting+(1-allocation_gate)*content_weighting)
        if (write_weighting.sum(1)>1).any() or (write_weighting<0).any():
            raise ValueError("write weighting simplex bound problem.")
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
        p_j=self.precedence_weighting.unsqueeze(1).expand(-1,param.N,-1)
        batch_temporal_memory_linkage=self.temporal_memory_linkage.expand(param.bs,-1,-1)

        self.temporal_memory_linkage= (1 - ww_j - ww_i) * batch_temporal_memory_linkage + ww_i * p_j
        return self.temporal_memory_linkage

    def backward_weighting(self):
        '''

        :return: backward_weighting: b^i_t, (N,R)
        '''
        return torch.matmul(self.temporal_memory_linkage, self.last_read_weightings)

    def forward_weighting(self):
        '''

        :return: forward_weighting: f^i_t, (N,R)
        '''
        return torch.matmul(self.temporal_memory_linkage.transpose(1,2), self.last_read_weightings)

    # TODO sparse update, skipped because it's for performance improvement.

    def read_weightings(self, forward_weighting, backward_weighting, read_keys,
                        read_strengths, read_modes):
        '''

        :param forward_weighting: (N,R)
        :param backward_weighting: (N,R)
        ****** read_content_weighting: C, (N,R), (0,1)
        :param read_keys: k^w_t, (W,R)
        :param read_key_strengths: (R)
        :param read_modes: /pi_t^i, (R,3)
        :return: read_weightings: w^r_t, (N,R)

        TODO how is there even a bug?
        read modes add up to 1
        all weightings are simplex bound. This is ridiculous.
        '''

        content_weighting=self.read_content_weighting(read_keys,read_strengths)
        # has dimension (3,N,R)
        all_weightings=torch.stack([backward_weighting,content_weighting,forward_weighting],dim=1)
        # permute to dimension (R,N,3)
        all_weightings=all_weightings.permute(0,3,2,1)
        # this is becuase torch.matmul is designed to iterate all dimension excluding the last two
        # dimension (R,3,1)
        read_modes=read_modes.unsqueeze(3)
        # dimension (N,R)
        read_weightings = torch.matmul(all_weightings, read_modes).squeeze(3).transpose(1,2)
        self.last_read_weightings=read_weightings
        # last read weightings
        if (self.last_read_weightings)
        return read_weightings

    def read_memory(self,read_weightings):
        '''

        memory: (N,W)
        read weightings: (N,R)

        :return: read_vectors: [r^i_R], (W,R)
        '''

        return torch.matmul(self.memory.t(),read_weightings)

    def write_to_memory(self,write_weighting,erase_vector,write_vector):
        '''

        :param write_weighting: the strength of writing
        :param erase_vector: e_t, (W), [0,1]
        :param write_vector: w^w_t, (W),
        TODO I am afraid that writing to memory would make batches processings
        interfere with each other
        :return:
        '''
        term1_2=torch.matmul(write_weighting.unsqueeze(2),erase_vector.unsqueeze(1))
        #problem code
        term1=self.memory.unsqueeze(0)*(torch.ones((param.bs,param.N,param.W))-term1_2)
        term2=torch.matmul(write_weighting.unsqueeze(2),write_vector.unsqueeze(1))
        self.memory=torch.mean(term1+term2, dim=0)
        if numpy.isinf(self.memory.detach().numpy()).any():
            raise ValueError("nan is found")
        self.previous_memory=self.memory

    def forward(self,read_keys, read_strengths, write_key, write_strength,
                erase_vector, write_vector, free_gates, allocation_gate,
                write_gate, read_modes):

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

        read_weightings=self.read_weightings(forward_weighting, backward_weighting, read_keys, read_strengths,
                                             read_modes)
        # read from memory last, a new modification.
        read_vectors=self.read_memory(read_weightings)

        return read_vectors