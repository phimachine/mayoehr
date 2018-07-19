# the goal of monster is to make sure that no graphs branch.
# this is so that our autograd works
# I have 80% confidence that this is a problem with 0.3.1
# I implemented it on 0.4 and I have never dealt with this atrocity.
import torch
from torch import nn
import pdb
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity, softmax, normalize
from torch.nn.parameter import Parameter
import math

debug = False


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


class Frankenstein(nn.Module):
    def __init__(self,
                 x=47782,
                 h=128,
                 L=16,
                 v_t=3656,
                 W=32,
                 R=16,
                 N=64,
                 bs=1):
        super(Frankenstein, self).__init__()

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
            self.RNN_list.append(RNN_Unit(self.x, self.R, self.W, self.h, self.bs))
        self.hidden_previous_timestep = Variable(torch.Tensor(self.bs, self.L, self.h).zero_().cuda())
        self.W_y = Parameter(torch.Tensor(self.L * self.h, self.v_t).cuda(), requires_grad=True)
        self.W_E = Parameter(torch.Tensor(self.L * self.h, self.E_t).cuda(), requires_grad=True)
        self.b_y = Parameter(torch.Tensor(self.v_t).cuda(), requires_grad=True)
        self.b_E = Parameter(torch.Tensor(self.E_t).cuda(), requires_grad=True)

        stdv = 1.0 / math.sqrt(self.v_t)
        self.W_y.data.uniform_(-stdv, stdv)
        self.b_y.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.E_t)
        self.W_E.data.uniform_(-stdv, stdv)
        self.b_E.data.uniform_(-stdv, stdv)

        '''MEMORY'''
        # u_0
        self.usage_vector = Variable(torch.Tensor(self.bs, self.N).zero_().cuda())
        # p, (N), should be simplex bound
        self.precedence_weighting = Variable(torch.Tensor(self.bs, self.N).zero_().cuda())
        # (N,N)
        self.temporal_memory_linkage = Variable(torch.Tensor(self.bs, self.N, self.N).zero_().cuda())
        # (N,W)
        self.memory = Variable(torch.Tensor(self.N, self.W).zero_().cuda())
        # (N, R).
        self.last_read_weightings = Variable(torch.Tensor(self.bs, self.N, self.R).fill_(1.0 / self.N)).cuda()

        '''COMPUTER'''
        # see paper, paragraph 2 page 7
        self.last_read_vector = Variable(torch.Tensor(self.bs, self.W, self.R).zero_().cuda())
        self.W_r = Parameter(torch.Tensor(self.W * self.R, self.v_t).cuda())

        stdv = 1.0 / math.sqrt(self.v_t)
        self.W_r.data.uniform_(-stdv, stdv)

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

        # (self.bs, self.N)
        innerprod = torch.matmul(write_key, self.memory.t())
        # (parm.N)
        memnorm = torch.norm(self.memory, 2, 1)
        # (self.bs)
        writenorm = torch.norm(write_key, 2, 1)
        # (self.N, self.bs)
        normalizer = torch.ger(memnorm, writenorm)
        similarties = innerprod / normalizer.t().clamp(min=eps)
        similarties = similarties * key_strength.expand(-1, self.N)
        normalized = softmax(similarties, dim=1)
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

        innerprod = torch.matmul(self.memory.unsqueeze(0), read_keys)
        # this is confusing. matrix[n] access nth row, not column
        # this is very counter-intuitive, since columns have meaning,
        # because they represent vectors
        mem_norm = torch.norm(self.memory, p=2, dim=1)
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

    def update_usage_vector(self, write_wighting, memory_retention):
        '''

        :param previous_usage: u_{t-1}, (N), [0,1]
        :param write_wighting: w^w_{t-1}, (N), simplex bound
        :param memory_retention: \psi_t, (N), simplex bound
        :return: u_t, (N), [0,1], the next usage,
        '''

        ret = (self.usage_vector + write_wighting - self.usage_vector * write_wighting) * memory_retention

        # Here we should use .data instead? Like:
        # self.usage_vector.data=ret.data
        # Usage vector contain all computation history,
        # which is not necessary? I'm not sure, maybe the write weighting should be back_propped here?
        # We reset usage vector for every seq, but should we for every timestep?
        self.usage_vector = ret
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
        sorted, indices = self.usage_vector.sort(dim=1)
        cum_prod = torch.cumprod(sorted, 1)
        # notice the index on the product
        cum_prod = torch.cat([Variable(torch.ones(self.bs, 1).cuda()), cum_prod], 1)[:, :-1]
        sorted_inv = 1 - sorted
        allocation_weighting = sorted_inv * cum_prod
        # to shuffle back in place
        ret = torch.gather(allocation_weighting, 1, indices)
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
        self.precedence_weighting = ((1 - sum_ww).unsqueeze(1) * self.precedence_weighting + write_weighting)
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
        batch_temporal_memory_linkage = self.temporal_memory_linkage.expand(self.bs, -1, -1)
        self.temporal_memory_linkage = ((1 - ww_j - ww_i) * batch_temporal_memory_linkage + ww_i * p_j)
        if debug:
            test_simplex_bound(self.temporal_memory_linkage, 1)
            test_simplex_bound(self.temporal_memory_linkage.transpose(1, 2), 1)
        return self.temporal_memory_linkage

    def backward_weighting(self):
        '''

        :return: backward_weighting: b^i_t, (N,R)
        '''
        ret = torch.matmul(self.temporal_memory_linkage, self.last_read_weightings)
        if debug:
            test_simplex_bound(ret, 1)
        return ret

    def forward_weighting(self):
        '''

        :return: forward_weighting: f^i_t, (N,R)
        '''
        ret = torch.matmul(self.temporal_memory_linkage.transpose(1, 2), self.last_read_weightings)
        if debug:
            test_simplex_bound(ret, 1)
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
        self.last_read_weightings = read_weightings
        # last read weightings
        if debug:
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

        return torch.matmul(self.memory.t(), read_weightings)

    def write_to_memory(self, write_weighting, erase_vector, write_vector):
        '''

        :param write_weighting: the strength of writing
        :param erase_vector: e_t, (W), [0,1]
        :param write_vector: w^w_t, (W),
        :return:
        '''
        term1_2 = torch.matmul(write_weighting.unsqueeze(2), erase_vector.unsqueeze(1))
        # term1=self.memory.unsqueeze(0)*Variable(torch.ones((self.bs,self.N,self.W)).cuda()-term1_2.data)
        term1 = self.memory.unsqueeze(0) * (1 - term1_2)
        term2 = torch.matmul(write_weighting.unsqueeze(2), write_vector.unsqueeze(1))
        self.memory = torch.mean(term1 + term2, dim=0)

    def reset_parameters(self):
        for module in self.RNN_list:
            # this should iterate over RNN_Units only
            module.reset_parameters()
        stdv = 1.0 / math.sqrt(self.v_t)
        self.W_y.data.uniform_(-stdv, stdv)
        self.b_y.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.E_t)
        self.W_E.data.uniform_(-stdv, stdv)
        self.b_E.data.uniform_(-stdv, stdv)

    def new_sequence_reset(self):
        '''controller'''
        self.hidden_previous_timestep = Variable(torch.Tensor(self.bs, self.L, self.h).zero_().cuda())
        for RNN in self.RNN_list:
            RNN.new_sequence_reset()
        self.W_y = Parameter(self.W_y.data, requires_grad=True)
        self.b_y = Parameter(self.b_y.data, requires_grad=True)
        self.W_E = Parameter(self.W_E.data, requires_grad=True)
        self.b_E = Parameter(self.b_E.data, requires_grad=True)

        '''memory'''
        self.memory = Variable(self.memory.data)
        self.usage_vector = Variable(torch.Tensor(self.bs, self.N).zero_().cuda())
        self.precedence_weighting = Variable(torch.Tensor(self.bs, self.N).zero_().cuda())
        self.temporal_memory_linkage = Variable(torch.Tensor(self.bs, self.N, self.N).zero_().cuda())
        self.last_read_weightings = Variable(torch.Tensor(self.bs, self.N, self.R).fill_(1.0 / self.N).cuda())

        '''computer'''
        self.last_read_vector = Variable(torch.Tensor(self.bs, self.W, self.R).zero_().cuda())
        self.W_r = Parameter(self.W_r.data)

    def forward(self, input):
        input_x_t = torch.cat((input, self.last_read_vector.view(self.bs, -1)), dim=1)

        '''Controller'''
        hidden_previous_layer = Variable(torch.Tensor(self.bs, self.h).zero_().cuda())
        hidden_this_timestep = Variable(torch.Tensor(self.bs, self.L, self.h).cuda())
        for i in range(self.L):
            hidden_output = self.RNN_list[i](input_x_t, self.hidden_previous_timestep[:, i, :],
                                             hidden_previous_layer)
            hidden_this_timestep[:, i, :] = hidden_output
            hidden_previous_layer = hidden_output

        flat_hidden = hidden_this_timestep.view((self.bs, self.L * self.h))
        output = torch.matmul(flat_hidden, self.W_y)
        interface_input = torch.matmul(flat_hidden, self.W_E)
        self.hidden_previous_timestep = hidden_this_timestep

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

        # R free gates? [R] TODO what is this?
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
        allocation_weighting = self.allocation_weighting()
        write_weighting = self.write_weighting(write_key, write_strength,
                                               allocation_gate, write_gate, allocation_weighting)
        self.write_to_memory(write_weighting, erase_vector, write_vector)
        # update some
        memory_retention = self.memory_retention(free_gates)
        self.update_usage_vector(write_weighting, memory_retention)
        self.update_temporal_linkage_matrix(write_weighting)
        self.update_precedence_weighting(write_weighting)

        forward_weighting = self.forward_weighting()
        backward_weighting = self.backward_weighting()

        read_weightings = self.read_weightings(forward_weighting, backward_weighting, read_keys, read_strengths,
                                               read_modes)
        # read from memory last, a new modification.
        self.last_read_vector = self.read_memory(read_weightings)

        if debug:
            if (self.last_read_vector != self.last_read_vector).any():
                raise ValueError("Nan is found")

        '''back to computer'''
        output2 = output + torch.matmul(self.last_read_vector.view(self.bs, self.W * self.R), self.W_r)

        # DEBUG NAN
        if (self.last_read_vector != self.last_read_vector).any():
            raise ValueError("nan is found.")
        if (output2 != output2).any():
            raise ValueError("nan is found.")
        return output2

# the problem seems to be in the RNN Unit.
# we exchange the RNN units and one has memory overflow, the other has convergence issues.

#### MEMORY OVERFLOW ISSUE
class RNN_Unit(nn.Module):
    """
    A single unit of deep RNN
    """

    def __init__(self, x, R, W, h, bs):
        super(RNN_Unit, self).__init__()

        self.x = x
        self.R = R
        self.W = W
        self.h = h
        self.bs = bs

        self.W_input = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
        self.W_forget = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
        self.W_output = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)
        self.W_state = nn.Linear(self.x + self.R * self.W + 2 * self.h, self.h)

        self.old_state = Variable(torch.Tensor(self.bs, self.h).zero_().cuda())

    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()

    def forward(self, input_x, previous_time, previous_layer):
        # a hidden unit outputs a hidden output new_hidden.
        # state also changes, but it's hidden inside a hidden unit.

        semicolon_input = torch.cat([input_x, previous_time, previous_layer], dim=1)

        # 5 equations
        input_gate = torch.sigmoid(self.W_input(semicolon_input))
        forget_gate = torch.sigmoid(self.W_forget(semicolon_input))
        new_state = forget_gate * self.old_state + input_gate * \
                    torch.tanh(self.W_state(semicolon_input))
        output_gate = torch.sigmoid(self.W_output(semicolon_input))
        new_hidden = output_gate * torch.tanh(new_state)
        self.old_state = new_state

        return new_hidden

    def new_sequence_reset(self):
        self.W_input.weight.detach()
        self.W_input.bias.detach()
        self.W_output.weight.detach()
        self.W_output.bias.detach()
        self.W_forget.weight.detach()
        self.W_forget.bias.detach()
        self.W_state.weight.detach()
        self.W_state.bias.detach()

        self.old_state = Variable(torch.Tensor(self.bs, self.h).zero_().cuda())

#### CONVERGENCE ISSUE
#
# class RNN_Unit(nn.Module):
#     """
#     LSTM
#     """
#
#     def __init__(self, x, R, W, h, bs):
#         super(RNN_Unit, self).__init__()
#
#         self.x = x
#         self.R = R
#         self.W = W
#         self.h = h
#         self.bs = bs
#
#         self.W_input = Parameter(torch.Tensor(self.x + self.R * self.W + 2 * self.h, self.h).cuda(), requires_grad=True)
#         self.W_forget = Parameter(torch.Tensor(self.x + self.R * self.W + 2 * self.h, self.h).cuda(),
#                                   requires_grad=True)
#         self.W_output = Parameter(torch.Tensor(self.x + self.R * self.W + 2 * self.h, self.h).cuda(),
#                                   requires_grad=True)
#         self.W_state = Parameter(torch.Tensor(self.x + self.R * self.W + 2 * self.h, self.h).cuda(), requires_grad=True)
#
#         self.b_input = Parameter(torch.Tensor(self.h).cuda(), requires_grad=True)
#         self.b_forget = Parameter(torch.Tensor(self.h).cuda(), requires_grad=True)
#         self.b_output = Parameter(torch.Tensor(self.h).cuda(), requires_grad=True)
#         self.b_state = Parameter(torch.Tensor(self.h).cuda(), requires_grad=True)
#
#         wls = (self.W_forget, self.W_output, self.W_input, self.W_state, self.b_input, self.b_forget, self.b_output,
#                self.b_state)
#         stdv = 1.0 / math.sqrt(self.h)
#         for w in wls:
#             w.data.uniform_(-stdv, stdv)
#
#         self.old_state = Variable(torch.Tensor(self.bs, self.h).zero_().cuda())
#     #
#     # def reset_parameters(self):
#     #     # initialized the way pytorch LSTM is initialized, from normal
#     #     # initial state and cell are empty
#     #
#     #     # if this is not run, any output might be nan
#     #     stdv = 1.0 / math.sqrt(self.h)
#     #     for weight in self.parameters():
#     #         weight.data.uniform_(-stdv, stdv)
#     #     for module in self.children():
#     #         # do not use self.modules(), because it would be recursive
#     #         module.reset_parameters()
#
#     def reset_parameters(self):
#         for module in self.children():
#             module.reset_parameters()
#
#     def forward(self, input_x, previous_time, previous_layer):
#         # a hidden unit outputs a hidden output new_hidden.
#         # state also changes, but it's hidden inside a hidden unit.
#
#         # I think .data call is safe whenever new Variable should be initated.
#         # Unless I wish to have the gradients recursively flow back to the beginning of history
#         # I do not wish so.
#         semicolon_input = torch.cat((input_x, previous_time, previous_layer), dim=1)
#
#         # 5 equations
#         input_gate = torch.sigmoid(torch.matmul(semicolon_input, self.W_input) + self.b_input)
#         forget_gate = torch.sigmoid(torch.matmul(semicolon_input, self.W_forget) + self.b_forget)
#         new_state = forget_gate * self.old_state + input_gate * \
#                     torch.tanh(torch.matmul(semicolon_input, self.W_state) + self.b_state)
#         output_gate = torch.sigmoid(torch.matmul(semicolon_input, self.W_output) + self.b_output)
#         new_hidden = output_gate * torch.tanh(new_state)
#
#         # TODO Warning: needs to assign, not sure if this is right
#         # Good warning, I have changed the assignment and I hope this now works better.
#         self.old_state = new_state
#
#         return new_hidden
#
#     def new_sequence_reset(self):
#         self.old_state = Variable(torch.Tensor(self.bs, self.h).zero_().cuda())
#         # self.W_input.weight.detach()
#         # self.W_input.bias.detach()
#         # self.W_forget.weight.detach()
#         # self.W_forget.bias.detach()
#         # self.W_output.weight.detach()
#         # self.W_output.bias.detach()
#         # self.W_state.weight.detach()
#         # self.W_state.bias.detach()
#         # self.old_state.detach()
#         self.W_input = Parameter(self.W_input.data, requires_grad=True)
#         self.W_forget = Parameter(self.W_forget.data, requires_grad=True)
#         self.W_output = Parameter(self.W_output.data, requires_grad=True)
#         self.W_state = Parameter(self.W_state.data, requires_grad=True)
#
#         self.b_input = Parameter(self.b_input.data, requires_grad=True)
#         self.b_forget = Parameter(self.b_forget.data, requires_grad=True)
#         self.b_output = Parameter(self.b_output.data, requires_grad=True)
#         self.b_state = Parameter(self.b_state.data, requires_grad=True)
