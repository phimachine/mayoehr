from archi.controller import RNN_Unit
from archi.computer import Computer
from archi.param import *
from archi.interface import Interface
from archi.controller import MyController
from archi.memory import Memory
from Monster import MonsterDNC
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import archi.param as param
import math
from torch.nn.modules.rnn import LSTM
from torch.autograd import Variable
# Lesson learnt:
# Use interactive prompt for debugging
# Then use script like this. Unit tests.


#
# hello=RNN_Unit()
# print(list(hello.parameters()))
#
# class hello2(nn.Module):
#     def __init__(self):
#         super(hello2, self).__init__()
#         self.haha=Parameter(torch.Tensor([1,2,3])).cuda()
#         self.haha.data.uniform_(-1,1)
#
#
# maybe=hello2()
# print(list(maybe.parameters()))

# apparently you cannot call cuda at parameter.

# hello=Computer()
# print(list(hello.children()))
# print(list(hello.parameters()))
#
# hello=MyController()
# optim=torch.optim.Adam(hello.parameters())
#
# for i in range(100):
#     x=Variable(torch.Tensor(1, param.x+param.R*param.W).uniform_(-1,1)).cuda()
#     output,interface=hello(x)
#     loss=output.sum()+interface.sum()
#     loss.backward()
#     optim.step()
#     hello.new_sequence_reset()


class memwrapper(nn.Module):
    def __init__(self):
        super(memwrapper, self).__init__()
        self.inter=Interface()
        self.mem=Memory()
        self.ctrl=MyController()
        self.W_r = Parameter(torch.Tensor(param.W * param.R, param.v_t).zero_().cuda())

    def forward(self, input):
        intout=self.inter(input)
        read_vec=self.mem(*intout)

        output = torch.matmul(read_vec.view(param.bs, param.W * param.R),self.W_r)

        return output

    def new_sequence_reset(self):
        self.W_r = Parameter(torch.Tensor(param.W * param.R, param.v_t).zero_().cuda())
        self.mem.new_sequence_reset()

# hello=memwrapper()
# optim=torch.optim.Adam(hello.parameters())
#
# for i in range(100):
#     x=Variable(torch.Tensor(1,param.E_t).uniform_(-1,1)).cuda()
#     ctrx = Variable(torch.Tensor(1, param.x + param.R * param.W).uniform_(-1, 1)).cuda()
#     output=hello(x)
#     loss=output.sum()
#     loss.backward()
#     optim.step()
#     hello.new_sequence_reset()
#     print("success")

class isadditiontheproblem(nn.Module):
    def __init__(self):
        super(isadditiontheproblem, self).__init__()
        self.inter=Interface()
        self.mem=Memory()
        self.ctrl=MyController()
        self.W_r = Parameter(torch.Tensor(param.W * param.R, param.v_t).zero_().cuda())

    def forward(self, input,ctrinput):
        output, _=self.ctrl(ctrinput)

        intout=self.inter(input)
        read_vec=self.mem(*intout)

        bbb=torch.matmul(read_vec.view(param.bs, param.W * param.R),self.W_r)
        output = output+bbb
        return output

    def new_sequence_reset(self):
        self.W_r = Parameter(torch.Tensor(param.W * param.R, param.v_t).zero_().cuda())
        self.mem.new_sequence_reset()
        self.ctrl.new_sequence_reset()

hello=isadditiontheproblem()
optim=torch.optim.Adam(hello.parameters())

# for i in range(100):
#     x=Variable(torch.Tensor(1,param.E_t).uniform_(-1,1)).cuda()
#     ctrx = Variable(torch.Tensor(1, param.x + param.R * param.W).uniform_(-1, 1)).cuda()
#     output=hello(x,ctrx)
#     loss=output.sum()
#     loss.backward()
#     optim.step()
#     hello.new_sequence_reset()
#     print("addition is not the problem")



class issharinginputtheproblem(nn.Module):
    def __init__(self):
        super(issharinginputtheproblem, self).__init__()
        self.inter=Interface()
        self.mem=Memory()
        self.ctrl=MyController()
        self.W_r = Parameter(torch.Tensor(param.W * param.R, param.v_t).zero_().cuda())

    def forward(self, ctrinput):
        output,intout=self.ctrl(ctrinput)

        intout=self.inter(intout)
        read_vec=self.mem(*intout)

        bbb=torch.matmul(read_vec.view(param.bs, param.W * param.R),self.W_r)
        output = output+bbb
        return output

    def new_sequence_reset(self):
        self.W_r = Parameter(torch.Tensor(param.W * param.R, param.v_t).zero_().cuda())
        self.mem.new_sequence_reset()
        self.ctrl.new_sequence_reset()

hello=issharinginputtheproblem()
optim=torch.optim.Adam(hello.parameters())
#
# for i in range(100):
#     x=Variable(torch.Tensor(1,param.E_t).uniform_(-1,1)).cuda()
#     ctrx = Variable(torch.Tensor(1, param.x + param.R * param.W).uniform_(-1, 1)).cuda()
#     output=hello(ctrx)
#     loss=output.sum()
#     loss.backward()
#     optim.step()
#     hello.new_sequence_reset()
#     print("what the heck")

hello=MonsterDNC()
optim=torch.optim.Adam(hello.parameters())

for i in range(100):
    x=Variable(torch.Tensor(1,param.E_t).uniform_(-1,1)).cuda()
    ctrx = Variable(torch.Tensor(1, param.x).uniform_(-1, 1)).cuda()
    output=hello(ctrx)
    loss=output.sum()
    loss.backward()
    optim.step()
    hello.new_sequence_reset()
    print("what the heck")


#
#
# class wrapper(nn.Module):
#     def __init__(self):
#         super(wrapper, self).__init__()
#         self.memory = Memory()
#         self.controller = MyController()
#         self.interface = Interface()
#         self.last_read_vector = Variable(torch.Tensor(param.bs, param.W, param.R).zero_().cuda())
#         # those work
#
#         self.W_r = Parameter(torch.Tensor(param.W * param.R, param.v_t).zero_().cuda())
#
#
#     def forward(self,input):
#         # fake a time-series. (bs, ts, ...)
#         # only applies to the off-the-shelf LSTM
#         # input_x_t=input_x_t.unsqueeze(1)
#         input_x_t = torch.cat((input, self.last_read_vector.view(param.bs, -1)), dim=1)
#
#         output, interface = self.controller(input_x_t)
#         interface_output_tuple = self.interface(interface)
#         self.last_read_vector = self.memory(*interface_output_tuple)
#         ######### TODO THIS LINE IS THE PROBLEM ################
#         # output = output + torch.matmul(self.last_read_vector.view(param.bs, param.W * param.R),self.W_r)
#
#         return output, self.last_read_vector
#
#     def new_sequence_reset(self):
#         # I have not found a reference to this function, but I think it's reasonable
#         # to reset the values that depends on a particular sequence.
#         self.controller.new_sequence_reset()
#         self.memory.new_sequence_reset()
#         self.last_read_vector = Variable(torch.Tensor(param.bs, param.W, param.R).zero_().cuda())
#         self.W_r = Parameter(torch.Tensor(param.W * param.R, param.v_t).zero_().cuda())
#
#
# hello=wrapper()
# optim=torch.optim.Adam(hello.parameters())
# for i in range(20):
#     x=Variable(torch.Tensor(1, param.x).uniform_(-1,1)).cuda()
#     output, lrv=hello(x)
#     loss=output.sum()+lrv.sum()
#     loss.backward()
#     optim.step()
#     hello.new_sequence_reset()



#
#
# hello=RNN_Unit()
# optim=torch.optim.Adam(hello.parameters())
# hidden_previous_timestep = Variable(torch.Tensor(param.bs, param.h).zero_().cuda())
# hidden_previous_layer = Variable(torch.Tensor(param.bs, param.h).zero_().cuda())
#
# for i in range(100):
#     x=Variable(torch.Tensor(1,param.x+param.R*param.W).uniform_(-1,1).cuda())
#     new_hidden=hello(x,hidden_previous_timestep,hidden_previous_layer)
#     loss=new_hidden.sum()
#     loss.backward()
#     optim.step()
#     hello.new_sequence_reset()

# hello=Computer()
# optim=torch.optim.Adam(hello.parameters())
# for i in range(100):
#     x=Variable(torch.Tensor(1,param.x).uniform_(-1,1).cuda())
#     output=hello(x)
#     loss=output.sum()
#     loss.backward()
#     optim.step()
#     hello.new_sequence_reset()


#
# lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
# inputs = [Variable(torch.randn(1, 3)) for _ in range(5)]  # make a sequence of length 5
#
# # initialize the hidden state.
# hidden = (Variable(torch.randn(1, 1, 3)),
#           Variable(torch.randn(1, 1, 3)))
# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     lstm.zero_grad()
#     out, hidden = lstm(i.view(1, 1, -1), hidden)
#     loss.backward()
#     optim.step()
