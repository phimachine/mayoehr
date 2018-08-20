"""
8/12/2018
Batch DNC is the model with externally stored states and batch processing timesteps on forward()
This architecture uses the stock LSTM.
"""

import torch
from torch import nn
import pdb
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity, softmax, normalize
from torch.nn.parameter import Parameter
import math
import numpy as np
import traceback
import os
from os.path import abspath
from pathlib import Path
import pickle
from torch.nn import LSTM
from death.DNC.bnpy import BatchNorm1d
debug = True

def sv(var):
    return var.data.cpu().numpy()

def test_simplex_bound(tensor, dim=1):
    # it's impossible to deal with dimensions
    # we will default to test dim 1 of 2-dim (x, y),
    # so that for every x, y is simplex bound

    if dim != 1:
        raise NotImplementedError("no longer accepts dim other othan one")
    t = tensor.contiguous()
    if (t.sum(1) - 1 > 1e-6).any() or (t.sum(1) < -1e-6).any() or (t < 0).any() or (t > 1).any():
        raise ValueError("test simplex bound failed")
    if (t != t).any():
        raise ValueError('test simple bound failed due to NA')
    return True

class BatchNorm(nn.Module):
    def __init__(self, channel_dim, eps=1e-5):
        super(BatchNorm, self).__init__()
        self.eps=Variable(torch.Tensor([eps])).cuda()
        self.lin=nn.Linear(channel_dim, channel_dim)

    def forward(self, input):
        # assume that the tensor is (batch, channels)
        mbmean=torch.mean(input, dim=0, keepdim=True)
        mbvar=torch.var(input, dim=0, keepdim=True)
        new_input=(input-mbmean)*torch.rsqrt(mbvar+self.eps)
        output=self.lin(new_input)

        return output

class BatchDNC(nn.Module):
    def __init__(self,
                 x=47764,
                 h=128,
                 L=16,
                 v_t=3620,
                 W=32,
                 R=8,
                 N=512,
                 bs=1):
        super(BatchDNC, self).__init__()

        # debugging usages
        self.last_state_dict=None

        '''PARAMETERS'''
        self.x = x
        self.h = h
        self.L = L
        self.v_t = v_t
        self.W = W
        self.R = R
        self.N = N
        self.bs = bs
        self.E_t = W * R + 3 * W + 5 * R + 3

        '''CONTROLLER'''
        # self.RNN_list = nn.ModuleList()
        # for _ in range(self.L):
        #     self.RNN_list.append(LSTM_Unit(self.x, self.R, self.W, self.h, self.bs))
        self.W_y = Parameter(torch.Tensor(self.L * self.h, self.v_t).cuda())
        self.W_E = Parameter(torch.Tensor(self.L * self.h, self.E_t).cuda())
        self.controller=Stock_LSTM(self.x, self.R, self.W, self.h, self.L, self.v_t)

        '''COMPUTER'''
        self.W_r = Parameter(torch.Tensor(self.W * self.R, self.v_t).cuda())
        # print("Using 0.4.1 PyTorch BatchNorm1d")
        self.bn = nn.BatchNorm1d(self.x, eps=1e-3, momentum=1e-10, affine=False)
        self.reset_parameters()

        '''States'''
        self.hidden_previous_timestep=None
        self.precedence_weighting=None
        self.temporal_memory_linkage=None
        self.memory=None
        self.last_read_weightings=None
        self.last_usage_vector=None
        self.last_write_weighting=None
        self.last_read_vector=None
        self.not_first_t_flag=None


    def init_states_each_channel(self):

        '''CONTROLLER'''

        '''MEMORY'''
        # p, (N), should be simplex bound
        precedence_weighting = Variable(torch.Tensor(1, self.N).cuda()).zero_()
        # (N,N)
        temporal_memory_linkage = Variable(torch.Tensor(1, self.N, self.N).cuda()).zero_()
        # (N,W)
        stdv = 1.0 / math.sqrt(self.W)
        memory = Variable(torch.Tensor(1, self.N, self.W).cuda().uniform_(-stdv,stdv))
        # (N, R).
        last_read_weightings = Variable(torch.Tensor(1, self.N, self.R).cuda()).zero_()
        # u_t, (N)
        last_usage_vector = Variable(torch.Tensor(1, self.N).cuda()).zero_()
        # store last write weightings for the calculation of usage vector
        last_write_weighting = Variable(torch.Tensor(1, self.N).cuda()).zero_()

        '''COMPUTER'''
        # needs to be initialized, otherwise throw NAN on first forward pass
        last_read_vector = Variable(torch.Tensor(1, self.W, self.R).cuda()).zero_()

        '''Second pass initiaion purpose'''
        not_first_t_flag = Variable(torch.Tensor(1, 1).cuda()).zero_().zero_()

        '''LSTM states'''
        # for lstm in self.RNN_list:
        #     states_list+=lstm.init_states_each_channel()
        h=Variable(torch.Tensor(self.L, 1, self.h)).zero_().cuda()
        c=Variable(torch.Tensor(self.L, 1, self.h)).zero_().cuda()

        states_tuple=(precedence_weighting, temporal_memory_linkage, memory,
                     last_read_weightings, last_usage_vector,last_write_weighting,
                     last_read_vector, not_first_t_flag, h, c)

        return states_tuple

    def reset_parameters(self):
        # if debug:
        #     print("parameters are reset")
        '''Controller'''
        # for module in self.RNN_list:
        #     # this should iterate over RNN_Units only
        #     module.reset_parameters()
        # self.hidden_previous_timestep.zero_()
        # # this breaks graph, only allowed on initializationf
        stdv = 1.0 / math.sqrt(self.v_t)
        self.W_y.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.E_t)
        self.W_E.data.uniform_(-stdv, stdv)
        self.controller.reset_parameters()

        '''Computer'''
        stdv = 1.0 / math.sqrt(self.v_t)
        self.W_r.data.uniform_(-stdv, stdv)

    def assign_states_tuple(self,states_tuple):
        '''
        This function needs to be called before every forward()

        :param states_tuple: packed state tuple
        :return:
        '''
        # at this point, all the state tuples must be concatenated on batch dimension
        # pass to self so that we don't need to modify other functions.

        self.precedence_weighting, self.temporal_memory_linkage, self.memory, \
        self.last_read_weightings, self.last_usage_vector, self.last_write_weighting, \
        self.last_read_vector, self.not_first_t_flag\
            = states_tuple[:8]
        # 9 is h 10 is c
        # each is (num_layers * num_directions, batch, hidden_size)
        self.controller.assign_states_tuple(states_tuple[8:10])
        # for i in range(self.L):
        #     i.assign_states_tuple(states_tuple[9+2*i:11+2*i])


    def forward(self, input):
        if (input!=input).any():
            raise ValueError("We have NAN in inputs")

        # dimension 42067 for all channels report NAN
        # train.mother.lab
        input=input.squeeze(1)

        try:
            bnout=self.bn(input)
            if (input != input).any():
                raise ValueError("We have NAN after batch normalization")
        except ValueError:
            print("reinitialize batch norm")
            self.bn = nn.BatchNorm1d(self.x, eps=1e-3, momentum=1e-10, affine=False)
            bnout=self.bn(input)

        input=bnout.unsqueeze(1)


        input_x_t = torch.cat((input, self.last_read_vector.view(self.bs, 1, -1)), dim=2)
        if (input_x_t!=input_x_t).any():
            raise ValueError("We have NAN in last read vector")
        '''Controller'''
        _, st=self.controller(input_x_t)
        if (input_x_t!=input_x_t).any():
            raise ValueError("We have NAN in LSTM outputs")
        h, c= st
        # was (num_layers, batch, hidden_size)
        hidden=h.permute(1,0,2)
        flat_hidden = hidden.contiguous().view((self.bs, self.L * self.h))
        vt = torch.matmul(flat_hidden, self.W_y)
        interface_input = torch.matmul(flat_hidden, self.W_E)
        # self.hidden_previous_timestep = h

        '''interface'''
        last_index = self.W * self.R

        # Read keys, each W dimensions, [W*R] in total
        # no processing needed
        # this is the address keys, not the contents
        read_keys = interface_input[:, 0:last_index].contiguous().view(self.bs, self.W, self.R)

        # Read strengths, [R]
        # 1 to infinity
        # slightly different equation from the paper, should be okay
        read_strengths = interface_input[:, last_index:last_index + self.R]
        last_index = last_index + self.R
        read_strengths = 1 - nn.functional.logsigmoid(read_strengths)

        # Write key, [W]
        write_key = interface_input[:, last_index:last_index + self.W]
        last_index = last_index + self.W

        # write strength beta, [1]
        write_strength = interface_input[:, last_index:last_index + 1]
        last_index = last_index + 1
        write_strength = 1 - nn.functional.logsigmoid(write_strength)

        # erase strength, [W]
        erase_vector = interface_input[:, last_index:last_index + self.W]
        last_index = last_index + self.W
        erase_vector = nn.functional.sigmoid(erase_vector)

        # write vector, [W]
        write_vector = interface_input[:, last_index:last_index + self.W]
        last_index = last_index + self.W

        # R free gates? [R]
        free_gates = interface_input[:, last_index:last_index + self.R]

        last_index = last_index + self.R
        free_gates = nn.functional.sigmoid(free_gates)

        # allocation gate [1]
        allocation_gate = interface_input[:, last_index:last_index + 1]
        last_index = last_index + 1
        allocation_gate = nn.functional.sigmoid(allocation_gate)

        # write gate [1]
        write_gate = interface_input[:, last_index:last_index + 1]
        last_index = last_index + 1
        write_gate = nn.functional.sigmoid(write_gate)

        # read modes [R,3]
        read_modes = interface_input[:, last_index:last_index + self.R * 3]
        read_modes = read_modes.contiguous().view(self.bs, self.R, 3)
        read_modes = nn.functional.softmax(read_modes,dim=2)

        '''memory'''
        memory_retention = self.memory_retention(free_gates)
        # usage vector update must be called before allocation weighting.
        self.update_usage_vector(memory_retention)
        allocation_weighting = self.allocation_weighting()

        write_weighting = self.write_weighting(write_key, write_strength,
                                               allocation_gate, write_gate, allocation_weighting)
        self.write_to_memory(write_weighting, erase_vector, write_vector)

        # update some
        self.update_temporal_linkage_matrix(write_weighting)
        self.update_precedence_weighting(write_weighting)

        forward_weighting = self.forward_weighting()
        backward_weighting = self.backward_weighting()

        read_weightings = self.read_weightings(forward_weighting, backward_weighting, read_keys, read_strengths,
                                               read_modes)
        # read from memory last, a new modification.
        read_vector = self.read_memory(read_weightings)
        # DEBUG NAN
        if (read_vector != read_vector).any():
            # this is a problem! TODO
            raise ValueError("nan is found.")

        '''back to computer'''
        yt = vt + torch.matmul(read_vector.view(self.bs, self.W * self.R), self.W_r)

        # update the last weightings
        self.last_read_vector=read_vector
        self.last_read_weightings = read_weightings
        self.last_write_weighting = write_weighting

        if debug:
            test_simplex_bound(self.last_read_weightings)
            test_simplex_bound(self.last_write_weighting)
            if (yt != yt).any():
                raise ValueError("nan is found.")

        states_tuple=(self.precedence_weighting, self.temporal_memory_linkage, \
                      self.memory, self.last_read_weightings, self.last_usage_vector, self.last_write_weighting, \
                      self.last_read_vector, self.not_first_t_flag, h, c)

        return yt, states_tuple

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

        write_key=write_key.unsqueeze(1)
        # (self.bs, self.N)
        innerprod = torch.matmul(write_key, self.memory.transpose(1,2)).squeeze(1)
        # (self.bs, parm.N)
        memnorm = torch.norm(self.memory, 2, 2)
        # (self.bs, 1)
        writenorm = torch.norm(write_key, 2, 2)
        # (self.bs, self.N)
        normalizer = memnorm*writenorm
        similarties = innerprod / normalizer.clamp(min=eps)
        similarties = similarties * key_strength.expand(-1, self.N)
        normalized = softmax(similarties, dim=1)
        if debug:
            if (normalized!=normalized).any():
                task_dir = os.path.dirname(abspath(__file__))
                save_dir = Path(task_dir) / "saves" / "keykey.pkl"
                pickle.dump((write_key.cpu(),key_strength.cpu()),save_dir.open('wb'))
                raise ValueError("NA found in write content weighting")
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

        # (bs, N, R)
        innerprod = torch.matmul(self.memory, read_keys)
        mem_norm = torch.norm(self.memory, p=2, dim=2)
        read_norm = torch.norm(read_keys, p=2, dim=1)
        mem_norm = mem_norm.unsqueeze(2)
        read_norm = read_norm.unsqueeze(1)
        # (bs, N, R)
        normalizer = torch.matmul(mem_norm, read_norm)

        # if transposed then similiarities[0] refers to the first read key
        similarties = innerprod / normalizer.clamp(min=eps)
        weighted = similarties * key_strengths.unsqueeze(1)
        ret = softmax(weighted, dim=1)
        # (bs, N, R)
        return ret

    # the highest freed will be retained? What does it mean?
    def memory_retention(self, free_gate):
        '''

        :param free_gate: f, (R), [0,1], from interface vector
        :param read_weighting: w^r_t, (N, R), simplex bounded,
               note it's from previous timestep.
        :return: \psi, (N), [0,1]
        '''

        # a free gate belongs to a read head.
        # a single read head weighting is a (N) dimensional simplex bounded value

        # (N, R)
        inside_bracket = 1 - self.last_read_weightings * free_gate.unsqueeze(1).expand(-1, self.N, -1)
        ret = torch.prod(inside_bracket, 2)
        return ret

    def update_usage_vector(self, memory_retention):
        '''

        :param memory_retention: \psi_t, (N), simplex bound
        :return: u_t, (N), [0,1], the next usage
        '''
        usage_vector = (self.last_usage_vector + self.last_write_weighting - self.last_usage_vector * self.last_write_weighting) \
              * memory_retention
        # if first_t, then usage_vector is asserted to be zero
        if (usage_vector != usage_vector).any():
            raise ValueError("NA found in usage vector, not first t flag reset behavior not certain")
        expandf = self.not_first_t_flag.expand(self.bs, self.N)
        usage_vector = usage_vector * expandf
        self.last_usage_vector=usage_vector
        return usage_vector

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

        # not the last usage, since we will update usage before this
        sorted, indices = self.last_usage_vector.sort(dim=1)
        cum_prod = torch.cumprod(sorted, 1)
        # notice the index on the product
        cum_prod = torch.cat([Variable(torch.ones(self.bs, 1).cuda()), cum_prod], 1)[:, :-1]
        sorted_inv = 1 - sorted
        allocation_weighting = sorted_inv * cum_prod
        # to shuffle back in place
        ret = torch.gather(allocation_weighting, 1, indices)
        if debug:
            if (ret!=ret).any():
                raise ValueError("NA found in allocation weighting")
        return ret

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
        content_weighting = self.write_content_weighting(write_key, write_strength)
        write_weighting = write_gate * (
                    allocation_gate * allocation_weighting + (1 - allocation_gate) * content_weighting)
        if debug:
            test_simplex_bound(write_weighting, 1)
        return write_weighting

    def update_precedence_weighting(self, write_weighting):
        '''

        :param write_weighting: (N)
        :return: self.precedence_weighting: (N), simplex bound
        '''
        # this is the bug. I called the python default sum() instead of torch.sum()
        # Took me 3 hours.
        # sum_ww=sum(write_weighting,1)
        sum_ww = torch.sum(write_weighting, dim=1)
        self.precedence_weighting = (1 - sum_ww).unsqueeze(1) * self.precedence_weighting + write_weighting
        if debug:
            test_simplex_bound(self.precedence_weighting, 1)
        return self.precedence_weighting

    def update_temporal_linkage_matrix(self, write_weighting):
        '''

        :param write_weighting: (N)
        :param precedence_weighting: (N), simplex bound
        :return: updated_temporal_linkage_matrix
        '''

        ww_j = write_weighting.unsqueeze(1).expand(-1, self.N, -1)
        ww_i = write_weighting.unsqueeze(2).expand(-1, -1, self.N)
        p_j = self.precedence_weighting.unsqueeze(1).expand(-1, self.N, -1)
        # batch_temporal_memory_linkage = self.temporal_memory_linkage.expand(-1, -1)
        newtml = (1 - ww_j - ww_i) * self.temporal_memory_linkage + ww_i * p_j
        is_cuda = ww_j.is_cuda
        if is_cuda:
            idx = torch.arange(0, self.N, out=torch.cuda.LongTensor())
        else:
            idx = torch.arange(0, self.N, out=torch.LongTensor())
        newtml[:, idx, idx] = 0
        if debug:
            try:
                test_simplex_bound(newtml, 1)
                test_simplex_bound(newtml.transpose(1, 2), 1)
            except ValueError:
                traceback.print_exc()
                print("precedence close to one?", precedence_weighting.sum() > 1)
                raise

        expandf = self.not_first_t_flag.unsqueeze(2).expand(self.bs, self.N, self.N)
        # force to be zero
        newtml = newtml * expandf
        self.temporal_memory_linkage=newtml

    def backward_weighting(self):
        '''
        :return: backward_weighting: b^i_t, (N,R)
        '''
        ret = torch.matmul(self.temporal_memory_linkage, self.last_read_weightings)
        if debug:
            test_simplex_bound(ret.permute(0,2,1), 1)
        return ret

    def forward_weighting(self):
        '''

        :return: forward_weighting: f^i_t, (N,R)
        '''
        ret = torch.matmul(self.temporal_memory_linkage.transpose(1, 2), self.last_read_weightings)
        if debug:
            test_simplex_bound(ret.permute(0,2,1))
        return ret

    # TODO sparse update, skipped because it's for performance improvement.

    def read_weightings(self, forward_weighting, backward_weighting, read_keys,
                        read_strengths, read_modes):
        '''

        :param forward_weighting: (bs,N,R)
        :param backward_weighting: (bs,N,R)
        ****** content_weighting: C, (bs,N,R), (0,1)
        :param read_keys: k^w_t, (bs,W,R)
        :param read_key_strengths: (bs,R)
        :param read_modes: /pi_t^i, (bs,R,3)
        :return: read_weightings: w^r_t, (bs,N,R)

        '''

        content_weighting = self.read_content_weighting(read_keys, read_strengths)
        if debug:
            test_simplex_bound(content_weighting, 1)
            test_simplex_bound(backward_weighting, 1)
            test_simplex_bound(forward_weighting, 1)
        # has dimension (bs,3,N,R)
        all_weightings = torch.stack([backward_weighting, content_weighting, forward_weighting], dim=1)
        # permute to dimension (bs,R,N,3)
        all_weightings = all_weightings.permute(0, 3, 2, 1)
        # this is becuase torch.matmul is designed to iterate all dimension excluding the last two
        # dimension (bs,R,3,1)
        read_modes = read_modes.unsqueeze(3)
        # dimension (bs,N,R)
        read_weightings = torch.matmul(all_weightings, read_modes).squeeze(3).transpose(1, 2)
        # last read weightings
        if debug:
            # if the second test passes, how come the first one does not?
            test_simplex_bound(self.last_read_weightings, 1)
            test_simplex_bound(read_weightings, 1)
            if (read_weightings != read_weightings).any():
                raise ValueError("NAN is found")
        return read_weightings

    def read_memory(self, read_weightings):
        '''

        memory: (N,W)
        read weightings: (N,R)

        :return: read_vectors: [r^i_R], (W,R)
        '''

        ret= torch.matmul(self.memory.transpose(1,2), read_weightings)
        return ret

    def write_to_memory(self, write_weighting, erase_vector, write_vector):
        '''

        :param write_weighting: the strength of writing
        :param erase_vector: e_t, (W), [0,1]
        :param write_vector: w^w_t, (W),
        :return:
        '''
        term1_2 = torch.matmul(write_weighting.unsqueeze(2), erase_vector.unsqueeze(1))
        # term1=self.memory.unsqueeze(0)*Variable(torch.ones((self.bs,self.N,self.W)).cuda()-term1_2.data)
        term1 = self.memory * (1 - term1_2)
        term2 = torch.matmul(write_weighting.unsqueeze(2), write_vector.unsqueeze(1))
        self.memory = term1 + term2


class Stock_LSTM(nn.Module):
    """
    I prefer using this Stock LSTM for numerical stability. The performance, however, is inferior.
    """
    def __init__(self, x, R, W, h, L, v_t):
        super(Stock_LSTM, self).__init__()

        self.x = x
        self.R = R
        self.W = W
        self.h = h
        self.L = L
        self.v_t= v_t

        self.LSTM=LSTM(input_size=self.x+self.R*self.W,hidden_size=h,num_layers=L,batch_first=True,
                       dropout=True)
        self.last=nn.Linear(self.h, self.v_t)
        self.st=None

    def forward(self, input_x):
        """
        :param input_x: input and memory values
        :return:
        """
        assert self.st is not None
        o, st = self.LSTM(input_x, self.st)
        if (st[0]!=st[0]).any():
            with open("debug/lstm.pkl") as f:
                pickle.dump(self, f)
            with open("debug/lstm.pkl") as f:
                pickle.dump(input_x, f)
            raise ("LSTM produced a NAN, objects dumped.")
        return self.last(o), st

    def reset_parameters(self):
        self.LSTM.reset_parameters()
        self.last.reset_parameters()

    def assign_states_tuple(self, states_tuple):
        self.st=states_tuple

# the problem seems to be in the RNN Unit.
# we exchange the RNN units and one has memory overflow, the other has convergence issues.
#
# class LSTM_Unit(nn.Module):
#     """
#     A single layer unit of LSTM
#     """
#
#     def __init__(self, x, R, W, h, bs):
#         super(LSTM_Unit, self).__init__()
#
#         self.x = x
#         self.R = R
#         self.W = W
#         self.h = h
#         self.bs = bs
#
#         self.W_input = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
#         self.W_forget = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
#         self.W_output = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
#         self.W_state = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
#
#         self.old_state = Variable(torch.Tensor(self.bs, self.h).zero_().cuda(),requires_grad=False)
#
#     def reset_parameters(self):
#         for module in self.children():
#             module.reset_parameters()
#
#     def forward(self, input_x, previous_time, previous_layer):
#         # a hidden unit outputs a hidden output new_hidden.
#         # state also changes, but it's hidden inside a hidden unit.
#
#         semicolon_input = torch.cat([input_x, previous_time, previous_layer], dim=1)
#
#         # 5 equations
#         input_gate = torch.sigmoid(self.W_input(semicolon_input))
#         forget_gate = torch.sigmoid(self.W_forget(semicolon_input))
#         new_state = forget_gate * self.old_state + input_gate * \
#                     torch.tanh(self.W_state(semicolon_input))
#         output_gate = torch.sigmoid(self.W_output(semicolon_input))
#         new_hidden = output_gate * torch.tanh(new_state)
#         self.old_state = Parameter(new_state.data,requires_grad=False)
#
#         return new_hidden
#
#
#     def reset_batch_channel(self,list_of_channels):
#         raise NotImplementedError()
#     #
#     # def new_sequence_reset(self):
#     #     raise DeprecationWarning("We no longer reset sequence together in all batch channels, this function deprecated")
#     #
#     #     self.W_input.weight.detach()
#     #     self.W_input.bias.detach()
#     #     self.W_output.weight.detach()
#     #     self.W_output.bias.detach()
#     #     self.W_forget.weight.detach()
#     #     self.W_forget.bias.detach()
#     #     self.W_state.weight.detach()
#     #     self.W_state.bias.detach()
#     #
#     #     self.old_state = Parameter(torch.Tensor(self.bs, self.h).zero_().cuda(),requires_grad=False)
