# the goal of monster is to make sure that no graphs branch.
# this is so that our autograd works
# I have 80% confidence that this is a problem with 0.3.1
# I implemented it on 0.4 and I have never dealt with this atrocity.
import torch
from torch import nn
import pdb
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

class MonsterDNC(nn.Module):
    def __init__(self,
                 x=47782,
                 h=2,
                 L=3,
                 v_t=3656,
                 W=4,
                 R=5,
                 N=6,
                 bs=1):
        super(MonsterDNC, self).__init__()

        '''CONTROLLER'''
        self.x=x
        self.h=h
        self.L=L
        self.v_t=v_t
        self.W=W
        self.R=R
        self.N=N
        self.bs=bs

        self.E_t = W * R + 3 * W + 5 * R + 3
        self.RNN_list=nn.ModuleList()
        for _ in range(self.L):
            self.RNN_list.append(RNN_Unit())
        self.hidden_previous_timestep=Variable(torch.Tensor(self.bs,self.L,self.h).zero_().cuda())
        # self.W_y=nn.Linear(self.L*self.h,self.v_t)
        # self.W_E=nn.Linear(self.L*self.h,self.E_t)
        self.W_y = Parameter(torch.Tensor(self.L * self.h, self.v_t).cuda(),requires_grad=True)
        self.W_E = Parameter(torch.Tensor(self.L * self.h, self.E_t).cuda(),requires_grad=True)
        self.b_y = Parameter(torch.Tensor(self.v_t).cuda(),requires_grad=True)
        self.b_E = Parameter(torch.Tensor(self.E_t).cuda(),requires_grad=True)

        wls=(self.W_y,self.W_E,self.b_y,self.b_E)
        stdv = 1.0 / math.sqrt(self.h)
        for w in wls:
            w.data.uniform_(-stdv, stdv)



class RNN_Unit(nn.Module):
    """
    LSTM
    """

    def __init__(self):
        super(RNN_Unit, self).__init__()
        # self.W_input=nn.Linear(self.x+self.R*self.W+2*self.h,self.h)
        # self.W_forget=nn.Linear(self.x+self.R*self.W+2*self.h,self.h)
        # self.W_output=nn.Linear(self.x+self.R*self.W+2*self.h,self.h)
        # self.W_state=nn.Linear(self.x+self.R*self.W+2*self.h,self.h)

        self.W_input=Parameter(torch.Tensor(self.x+self.R*self.W+2*self.h,self.h).cuda(),requires_grad=True)
        self.W_forget=Parameter(torch.Tensor(self.x+self.R*self.W+2*self.h,self.h).cuda(),requires_grad=True)
        self.W_output=Parameter(torch.Tensor(self.x+self.R*self.W+2*self.h,self.h).cuda(),requires_grad=True)
        self.W_state=Parameter(torch.Tensor(self.x+self.R*self.W+2*self.h,self.h).cuda(),requires_grad=True)

        self.b_input=Parameter(torch.Tensor(self.h).cuda(),requires_grad=True)
        self.b_forget=Parameter(torch.Tensor(self.h).cuda(),requires_grad=True)
        self.b_output=Parameter(torch.Tensor(self.h).cuda(),requires_grad=True)
        self.b_state=Parameter(torch.Tensor(self.h).cuda(),requires_grad=True)

        wls=(self.W_forget,self.W_output,self.W_input,self.W_state,self.b_input,self.b_forget,self.b_output,self.b_state)
        stdv = 1.0 / math.sqrt(self.h)
        for w in wls:
            w.data.uniform_(-stdv, stdv)

        self.old_state=Variable(torch.Tensor(self.bs,self.h).zero_().cuda())


    def reset_parameters(self):
        # initialized the way pytorch LSTM is initialized, from normal
        # initial state and cell are empty

        # if this is not run, any output might be nan
        stdv= 1.0 /math.sqrt(self.h)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
        for module in self.children():
            # do not use self.modules(), because it would be recursive
            module.reset_parameters()


    def forward(self,input_x,previous_time,previous_layer):
        # a hidden unit outputs a hidden output new_hidden.
        # state also changes, but it's hidden inside a hidden unit.

        # I think .data call is safe whenever new Variable should be initated.
        # Unless I wish to have the gradients recursively flow back to the beginning of history
        # I do not wish so.
        semicolon_input=torch.cat((input_x,previous_time,previous_layer),dim=1)

        # 5 equations
        input_gate=torch.sigmoid(torch.matmul(semicolon_input, self.W_input)+self.b_input)
        forget_gate=torch.sigmoid(torch.matmul(semicolon_input, self.W_forget)+self.b_forget)
        new_state=forget_gate * self.old_state + input_gate * \
                  torch.tanh(torch.matmul(semicolon_input, self.W_state)+self.b_state)
        output_gate=torch.sigmoid(torch.matmul(semicolon_input, self.W_output)+self.b_output)
        new_hidden=output_gate*torch.tanh(new_state)

        # TODO Warning: needs to assign, not sure if this is right
        # Good warning, I have changed the assignment and I hope this now works better.
        self.old_state=new_state

        return new_hidden

    def new_sequence_reset(self):
        self.old_state=Variable(torch.Tensor(self.bs,self.h).zero_().cuda())
        # self.W_input.weight.detach()
        # self.W_input.bias.detach()
        # self.W_forget.weight.detach()
        # self.W_forget.bias.detach()
        # self.W_output.weight.detach()
        # self.W_output.bias.detach()
        # self.W_state.weight.detach()
        # self.W_state.bias.detach()
        # self.old_state.detach()
        self.W_input= Parameter(self.W_input.data,requires_grad=True)
        self.W_forget=Parameter(self.W_forget.data,requires_grad=True)
        self.W_output=Parameter(self.W_output.data,requires_grad=True)
        self.W_state=Parameter(self.W_state.data,requires_grad=True)

        self.b_input= Parameter(self.b_input.data,requires_grad=True)
        self.b_forget=Parameter(self.b_forget.data,requires_grad=True)
        self.b_output=Parameter(self.b_output.data,requires_grad=True)
        self.b_state=Parameter(self.b_state.data,requires_grad=True)

        print('reset controller LSTM')
