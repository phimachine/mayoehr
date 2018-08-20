'''
Takes a state given from calling function scope
Big change.
This is a DNC that resets the experience at each new sequence.
'''

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

debug = True


def sv(var):
    return var.data.cpu().numpy()


def test_simplex_bound(tensor, dim=1):
    # it's impossible to deal with dimensions
    # we will default to test dim 1 of 2-dim (x, y),
    # so that for every x, y is simplex bound

    if dim != 1:
        raise DeprecationWarning("no longer accepts dim other othan one")
        raise NotImplementedError
    t = tensor.contiguous()
    if (t.sum(1) - 1 > 1e-6).any() or (t.sum(1) < -1e-6).any() or (t < 0).any() or (t > 1).any():
        raise ValueError("test simplex bound failed")
    if (t != t).any():
        raise ValueError('test simple bound failed due to NA')
    return True


class ResetDNC(nn.Module):
    def __init__(self,
                 x=47764,
                 h=128,
                 L=16,
                 v_t=3620,
                 W=32,
                 R=8,
                 N=512,
                 bs=1):
        super(ResetDNC, self).__init__()

        # debugging usages
        self.last_state_dict = None

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
        self.RNN_list = nn.ModuleList()
        for _ in range(self.L):
            self.RNN_list.append(LSTM_Unit(self.x, self.R, self.W, self.h, self.bs))
        self.hidden_previous_timestep = Variable(torch.Tensor(self.L, self.h).cuda())
        self.W_y = Parameter(torch.Tensor(self.L * self.h, self.v_t).cuda())
        self.W_E = Parameter(torch.Tensor(self.L * self.h, self.E_t).cuda())
        self.b_y = Parameter(torch.Tensor(self.v_t).cuda())
        self.b_E = Parameter(torch.Tensor(self.E_t).cuda())

        # '''MEMORY'''
        # self.precedence_weighting = Variable(torch.Tensor(self.N).cuda())
        # # (N,N)
        # self.temporal_memory_linkage = Variable(torch.Tensor(self.N, self.N)).cuda()
        # # (N,W)
        # # TODO NEEDS TO CHANGE ANY LOGIC THAT PROCESSES MEMORY
        # self.memory = Variable(torch.Tensor(self.N, self.W).cuda())
        # # (N, R).
        # self.last_read_weightings = Variable(torch.Tensor(self.N, self.R).cuda())
        # # u_t, (N)
        # self.last_usage_vector = Variable(torch.Tensor(self.N).cuda())
        # # store last write weightings for the calculation of usage vector
        # self.last_write_weighting = Variable(torch.Tensor(self.N)).cuda()
        #
        # self.first_t_flag = True

        '''COMPUTER'''
        # self.last_read_vector = Variable(torch.Tensor(self.W, self.R).cuda())
        self.W_r = Parameter(torch.Tensor(self.W * self.R, self.v_t).cuda())

        self.reset_parameters()

    def reset_parameters(self):
        # if debug:
        #     print("parameters are reset")
        '''Controller'''
        for module in self.RNN_list:
            # this should iterate over RNN_Units only
            module.reset_parameters()
        # self.hidden_previous_timestep.zero_()
        # self.controller_states_tuple=None
        stdv = 1.0 / math.sqrt(self.v_t)
        self.W_y.data.uniform_(-stdv, stdv)
        self.b_y.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.E_t)
        self.W_E.data.uniform_(-stdv, stdv)
        self.b_E.data.uniform_(-stdv, stdv)

        # '''Memory'''
        # self.precedence_weighting.zero_()
        # self.last_usage_vector.zero_()
        # self.last_read_weightings.zero_()
        # self.last_write_weighting.zero_()
        # self.temporal_memory_linkage.zero_()
        # # memory must be initialized like this, otherwise usage vector will be stuck at zero.
        # stdv = 1.0
        # self.memory.data.uniform_(-stdv, stdv)
        # self.first_t_flag = True

        '''Computer'''
        # see paper, paragraph 2 page 7
        # self.last_read_vector.zero_()
        stdv = 1.0 / math.sqrt(self.v_t)
        self.W_r.data.uniform_(-stdv, stdv)

    def new_states_tuple(self):
        """
        These variables will be stored in the calling function scope, for batch processing purposes
        When a new sequence starts in a channel, the variable on the corresponding dimension should be reinitialize
        Note that the parameters in a module does not need to be reinitialized
        :return:
        """

        # p, (N), should be simplex bound
        precedence_weighting = Variable(torch.Tensor(self.N).cuda()).zero_()
        # (N,N)
        temporal_memory_linkage = Variable(torch.Tensor(self.N, self.N)).cuda().zero_()
        # (N,W)
        stdv = 1.0 / math.sqrt(self.W)
        memory = Variable(torch.Tensor(self.N, self.W).cuda()).uniform(-stdv, stdv)
        # (N, R).
        last_read_weightings = Variable(torch.Tensor(self.N, self.R).cuda()).zero_()
        # u_t, (N)
        last_usage_vector = Variable(torch.Tensor(self.N).cuda()).zero_()
        # store last write weightings for the calculation of usage vector
        last_write_weighting = Variable(torch.Tensor(self.N)).cuda().zero_()
        not_first_t_flag = Varialbe(torch.Tensor(1).cuda()).zero_()
        last_read_vector = Variable(torch.Tensor(self.W, self.R).cuda()).zero_()

        # controller LSTM states
        controller_hidden = Variable(torch.Tensor(self.L, self.h).cuda()).zero_()
        controller_state = Variable(torch.Tensor(self.L, self.h).cuda().zero_())

        return precedence_weighting, temporal_memory_linkage, memory, last_read_weightings, last_usage_vector, \
               last_write_weighting, not_first_t_flag, last_read_vector, controller_hidden, controller_state

    def new_sequence_reset(self):
        '''
        The biggest question is whether to reset memory every time a new sequence is taken in.
        My take is to not reset the memory, but this might not be the best strategy there is.
        If memory is not reset at each new sequence, then we should not reset the memory at all?
        :return:
        '''

        # TODO see if this function is deprecated: is detach() necessary?
        # if debug:
        #     print('new sequence reset')
        '''controller'''
        for RNN in self.RNN_list:
            RNN.new_sequence_reset()
        self.W_y = Parameter(self.W_y.data)
        self.b_y = Parameter(self.b_y.data)
        self.W_E = Parameter(self.W_E.data)
        self.b_E = Parameter(self.b_E.data)

        '''memory'''

        # if self.reset:
        #     if self.palette:
        #         self.memory = Variable(initialz)
        #     else:
        #         # we will reset the memory altogether.
        #         # TODO The question is, should we reset the memory to a fixed state? There are good arguments for it.
        #         stdv = 1.0
        #         # gradient should not carry over, since at this stage, requires_grad on this parameter should be False.
        #         self.memory = Variable(torch.Tensor(self.N, self.W).cuda().uniform_(-stdv, stdv))
        #         # TODO is there a reason to reinitialize the parameter object? I don't think so. The graph is not carried over.
        #     #
        #     # self.last_usage_vector.zero_()
        #     # self.precedence_weighting.zero_()
        #     # self.temporal_memory_linkage.zero_()
        #     # self.last_read_weightings.zero_()
        #     # self.last_write_weighting.zero_()
        #
        #     self.last_usage_vector = Variable(torch.Tensor(self.N).zero_().cuda())
        #     self.precedence_weighting = Variable(torch.Tensor(self.N).zero_().cuda())
        #     self.temporal_memory_linkage = Variable(torch.Tensor(self.N, self.N).zero_().cuda())
        #     self.last_read_weightings = Variable(torch.Tensor(self.N, self.R).zero_().cuda())
        #     self.last_write_weighting = Variable(torch.Tensor(self.N).zero_().cuda())
        #
        # self.first_t_flag = True

        '''computer'''
        # self.last_read_vector = Variable(torch.Tensor(self.W, self.R).zero_().cuda())
        self.W_r = Parameter(self.W_r.data)

    def forward(self, input, states_tuple):
        # unpack states_tuple
        last_precedence_weighting, temporal_memory_linkage, last_memory, last_read_weightings, last_usage_vector, \
        last_write_weighting, not_first_t_flag, last_read_vector, controller_hidden, controller_state \
            = states_tuple

        if (input != input).any():
            raise ValueError("We have NAN in inputs")
        input_x_t = torch.cat((input, last_read_vector.view(-1)), dim=1)

        '''Controller'''
        hidden_previous_layer = Variable(torch.Tensor(self.h).zero_().cuda())
        new_controller_hidden = Variable(torch.Tensor(self.L, self.h).cuda())
        new_controller_state = Variable(torch.Tensor(self.L, self.h).cuda())
        for i in range(self.L):
            hidden_output, state_output = self.RNN_list[i](input_x_t, controller_hidden[:, i, :],
                                                           hidden_previous_layer, controller_state[:, i, :])
            if (hidden_output != hidden_output).any() or (state_output != state_output).any():
                raise ValueError("We have NAN in controller output.")
            new_controller_state[:, i, :] = state_output
            new_controller_hidden[:, i, :] = hidden_output
            hidden_previous_layer = hidden_output

        flat_hidden = hidden_this_timestep.view((self.L * self.h))
        output = torch.matmul(flat_hidden, self.W_y)
        interface_input = torch.matmul(flat_hidden, self.W_E)
        # needs to output
        # self.controller_hidden_previous_timestep = hidden_this_timestep
        # self.controller_state_previous_timestep = state_this_timestep

        '''interface'''
        last_index = self.W * self.R

        # Read keys, each W dimensions, [W*R] in total
        # no processing needed
        # this is the address keys, not the contents
        read_keys = interface_input[:, 0:last_index].contiguous().view(self.W, self.R)

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
        read_modes = read_modes.contiguous().view(self.R, 3)
        read_modes = nn.functional.softmax(read_modes, dim=2)

        '''memory'''
        memory_retention = self.memory_retention(free_gates, last_read_weightings)
        # usage vector update must be called before allocation weighting.
        new_usage_vector=self.usage_vector(memory_retention,last_write_weighting,
                                       memory_retention, not_first_t_flag)
        allocation_weighting = self.allocation_weighting(new_usage_vector)

        write_weighting = self.write_weighting(last_memory, write_key, write_strength,
                                               allocation_gate, write_gate, allocation_weighting)
        memory=self.new_memory(last_memory, write_weighting, erase_vector, write_vector)

        # update some
        newtlm=self.temporal_linkage_matrix(write_weighting, last_precedence_weighting, lasttml, not_first_t_flag)
        precedence_weighting=self.precedence_weighting(last_precedence_weighting,write_weighting)

        forward_weighting = self_weighting(newtml, last_read_weightings)
        backward_weighting = self.backward_weighting(newtml, last_read_weightings)

        read_weightings = self.read_weightings(memory, last_read_weightings, forward_weighting,
                                               backward_weighting, read_keys, read_strengths, read_modes)
        # read from memory last, a new modification.
        read_vector = self.read_memory(memory, read_weightings)
        # DEBUG NAN
        if (read_vector != read_vector).any():
            # this is a problem! TODO
            raise ValueError("nan is found.")

        '''back to computer'''
        output2 = output + torch.matmul(read_vector.view(self.W * self.R), self.W_r)

        # update the last weightings
        self.last_read_vector = read_vector
        self.last_read_weightings = read_weightings
        self.last_write_weighting = write_weighting

        self.first_t_flag = False

        if debug:
            test_simplex_bound(self.last_read_weightings)
            test_simplex_bound(self.last_write_weighting)
            if (output2 != output2).any():
                raise ValueError("nan is found.")

        return output2

    """
    All batch dimensions are implicit in following function documentations.
    """

    def write_content_weighting(self, memory, write_key, key_strength, eps=1e-8):
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

        # (self.N)
        innerprod = torch.matmul(write_key, memory.t())
        # (parm.N)
        memnorm = torch.norm(memory, 2, 1)
        # (self.bs)
        writenorm = torch.norm(write_key, 2, 1)
        # (self.N, self.bs)
        normalizer = torch.ger(memnorm, writenorm)
        similarties = innerprod / normalizer.t().clamp(min=eps)
        similarties = similarties * key_strength.expand(-1, self.N)
        normalized = softmax(similarties, dim=1)
        if debug:
            if (normalized != normalized).any():
                task_dir = os.path.dirname(abspath(__file__))
                save_dir = Path(task_dir) / "saves" / "keykey.pkl"
                pickle.dump((write_key.cpu(), key_strength.cpu()), save_dir.open('wb'))
                raise ValueError("NA found in write content weighting")
        return normalized

    def read_content_weighting(self, memory, read_keys, key_strengths, eps=1e-8):
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

        innerprod = torch.matmul(memory.unsqueeze(0), read_keys)
        # this is confusing. matrix[n] access nth row, not column
        # this is very counter-intuitive, since columns have meaning,
        # because they represent vectors
        mem_norm = torch.norm(memory, p=2, dim=1)
        read_norm = torch.norm(read_keys, p=2, dim=1)
        mem_norm = mem_norm.unsqueeze(1)
        read_norm = read_norm.unsqueeze(1)
        # (batch_size, locations, read_heads)
        normalizer = torch.matmul(mem_norm, read_norm)

        # if transposed then similiarities[0] refers to the first read key
        similarties = innerprod / normalizer.clamp(min=eps)
        weighted = similarties * key_strengths.unsqueeze(1).expand(-1, self.N, -1)
        ret = softmax(weighted, dim=1)
        return ret

    # the highest freed will be retained? What does it mean?
    def memory_retention(self, free_gate, last_read_weightings):
        '''

        :param free_gate: f, (R), [0,1], from interface vector
        :param read_weighting: w^r_t, (N, R), simplex bounded,
               note it's from previous timestep.
        :return: \psi, (N), [0,1]
        '''

        # a free gate belongs to a read head.
        # a single read head weighting is a (N) dimensional simplex bounded value

        # (N, R)
        inside_bracket = 1 - last_read_weightings * free_gate.unsqueeze(1).expand(-1, self.N, -1)
        ret = torch.prod(inside_bracket, 2)
        return ret

    def usage_vector(self, last_usage_vector, last_write_weighting,
                     memory_retention, not_first_t_flag):
        '''

        :param memory_retention: \psi_t, (N), simplex bound
        :return: u_t, (N), [0,1], the next usage
        '''
        usage_vector = (last_usage_vector + last_write_weighting - last_usage_vector * last_write_weighting) \
                       * memory_retention
        # if first_t, then usage_vector is asserted to be zero
        if (usage_vector != usage_vector).any():
            raise ValueError("NA found in usage vector, not first t flag reset behavior not certain")
        expandf = not_first_t_flag.expand(self.bs, self.N)
        usage_vector = usage_vector * not_first_t_flag
        return usage_vector

    def allocation_weighting(self, usage_vector):
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
        sorted, indices = usage_vector.sort(dim=1)
        cum_prod = torch.cumprod(sorted, 1)
        # notice the index on the product
        cum_prod = torch.cat([Variable(torch.ones(1).cuda()), cum_prod], 1)[:, :-1]
        sorted_inv = 1 - sorted
        allocation_weighting = sorted_inv * cum_prod
        # to shuffle back in place
        ret = torch.gather(allocation_weighting, 1, indices)
        if debug:
            if (ret != ret).any():
                raise ValueError("NA found in allocation weighting")
        return ret

    def write_weighting(self, memory, write_key, write_strength, allocation_gate, write_gate, allocation_weighting):
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
        content_weighting = self.write_content_weighting(memory, write_key, write_strength)
        write_weighting = write_gate * (
                allocation_gate * allocation_weighting + (1 - allocation_gate) * content_weighting)
        if debug:
            test_simplex_bound(write_weighting, 1)
        return write_weighting

    def precedence_weighting(self, last_precedence_weighting, write_weighting):
        '''

        :param write_weighting: (N)
        :return: new_precedence_weighting: (N), simplex bound
        '''
        # this is the bug. I called the python default sum() instead of torch.sum()
        # Took me 3 hours.
        # sum_ww=sum(write_weighting,1)
        sum_ww = torch.sum(write_weighting, dim=1)
        new_precedence_weighting = (1 - sum_ww).unsqueeze(1) * last_precedence_weighting + write_weighting
        if debug:
            test_simplex_bound(new_precedence_weighting, 1)
        return new_precedence_weighting

    def temporal_linkage_matrix(self, write_weighting, precedence_weighting, lasttml, not_first_t_flag):
        '''
        It's pretty hard to incorporate the logic of first_t_flag in it. A big question is whether NA*0 behaves.

        :param write_weighting: (N)
        :param precedence_weighting: (N), simplex bound
        :return: updated_temporal_linkage_matrix
        '''

        ww_j = write_weighting.unsqueeze(1).expand(-1, self.N, -1)
        ww_i = write_weighting.unsqueeze(2).expand(-1, -1, self.N)
        p_j = precedence_weighting.unsqueeze(1).expand(-1, self.N, -1)
        raise NotImplementedError("What the heck is this line?")
        batch_temporal_memory_linkage = lasttml.expand(-1, -1)
        newtml = (1 - ww_j - ww_i) * batch_temporal_memory_linkage + ww_i * p_j
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

        expandf = not_first_t_flag.expand(self, bs, self.N, self.N)
        # force to be zero
        newtml = newtml * expandf

        return newtml

    def backward_weighting(self, temporal_memory_linkage, last_read_weightings):
        '''
        :return: backward_weighting: b^i_t, (N,R)
        '''
        ret = torch.matmul(temporal_memory_linkage, last_read_weightings)
        if debug:
            test_simplex_bound(ret, 1)
        return ret

    def forward_weighting(self, temporal_memory_linkage, last_read_weightings):
        '''

        :return: forward_weighting: f^i_t, (N,R)
        '''
        ret = torch.matmul(temporal_memory_linkage.transpose(1, 2), last_read_weightings)
        if debug:
            test_simplex_bound(ret, 1)
        return ret

    # TODO sparse update, skipped because it's for performance improvement.

    def read_weightings(self, memory, last_read_weightings, forward_weighting, backward_weighting, read_keys,
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

        content_weighting = self.read_content_weighting(memory,read_keys, read_strengths)
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
            test_simplex_bound(last_read_weightings, 1)
            test_simplex_bound(read_weightings, 1)
            if (read_weightings != read_weightings).any():
                raise ValueError("NAN is found")
        return read_weightings

    def read_memory(self, memory, read_weightings):
        '''

        memory: (N,W)
        read weightings: (N,R)

        :return: read_vectors: [r^i_R], (W,R)
        '''

        return torch.matmul(memory.t(), read_weightings)

    def new_memory(self, last_memory, write_weighting, erase_vector, write_vector):
        '''

        :param write_weighting: the strength of writing
        :param erase_vector: e_t, (W), [0,1]
        :param write_vector: w^w_t, (W),
        :return:
        '''
        term1_2 = torch.matmul(write_weighting.unsqueeze(2), erase_vector.unsqueeze(1))
        # term1=self.memory.unsqueeze(0)*Variable(torch.ones((self.N,self.W)).cuda()-term1_2.data)
        term1 = last_memory.unsqueeze(0) * (1 - term1_2)
        term2 = torch.matmul(write_weighting.unsqueeze(2), write_vector.unsqueeze(1))
        return torch.mean(term1 + term2, dim=0)


class LSTM_Unit(nn.Module):
    """
    A single layer unit of LSTM
    """

    def __init__(self, x, R, W, h, bs):
        super(LSTM_Unit, self).__init__()

        self.x = x
        self.R = R
        self.W = W
        self.h = h
        self.bs = bs

        self.W_input = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
        self.W_forget = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
        self.W_output = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
        self.W_state = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)

        # self.old_state = Parameter(torch.Tensor(self.h).zero_().cuda(), requires_grad=False)

    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()

    def forward(self, input_x, previous_time, previous_layer, old_state):
        # a hidden unit outputs a hidden output new_hidden.
        # state also changes, but it's hidden inside a hidden unit.

        semicolon_input = torch.cat([input_x, previous_time, previous_layer], dim=1)

        # 5 equations
        input_gate = torch.sigmoid(self.W_input(semicolon_input))
        forget_gate = torch.sigmoid(self.W_forget(semicolon_input))
        new_state = forget_gate * old_state + input_gate * \
                    torch.tanh(self.W_state(semicolon_input))
        output_gate = torch.sigmoid(self.W_output(semicolon_input))
        new_hidden = output_gate * torch.tanh(new_state)

        return new_hidden, new_state

    def new_sequence_reset(self):
        # TODO I should experiment and see if detach has toi be called here.

        # is detach in place?
        self.W_input.weight.detach()
        self.W_input.bias.detach()
        self.W_output.weight.detach()
        self.W_output.bias.detach()
        self.W_forget.weight.detach()
        self.W_forget.bias.detach()
        self.W_state.weight.detach()
        self.W_state.bias.detach()

        # self.old_state = Parameter(torch.Tensor(self.h).zero_().cuda(), requires_grad=False)
