import torch
from torch.nn import LSTM
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter


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

        self.W_input = nn.Linear(self.x, self.h)
        self.W_forget = nn.Linear(self.x, self.h)
        self.W_output = nn.Linear(self.x, self.h)
        self.W_state = nn.Linear(self.x, self.h)

        self.old_state = Variable(torch.Tensor(self.bs, self.h).zero_().cuda(),requires_grad=False)

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
        self.old_state = Parameter(new_state.data,requires_grad=False)

        return new_hidden


    def reset_batch_channel(self,list_of_channels):
        raise NotImplementedError()

    def new_sequence_reset(self):
        raise DeprecationWarning("We no longer reset sequence together in all batch channels, this function deprecated")

        self.W_input.weight.detach()
        self.W_input.bias.detach()
        self.W_output.weight.detach()
        self.W_output.bias.detach()
        self.W_forget.weight.detach()
        self.W_forget.bias.detach()
        self.W_state.weight.detach()
        self.W_state.bias.detach()

        self.old_state = Parameter(torch.Tensor(self.bs, self.h).zero_().cuda(),requires_grad=False)

class LayeredLSTM(nn.Module):
    def __init__(self,
                 x=47764,
                 h=128,
                 L=16,
                 v_t=3620,
                 W=32,
                 R=8,
                 N=512,
                 bs=1):
        super(LayeredLSTM, self).__init__()

        self.x=x
        self.R=R
        self.W=W
        self.L=L
        self.h=h
        self.v_t=v_t
        self.bs=bs

        self.W_y = Parameter(torch.Tensor(self.L * self.h, self.v_t).cuda())
        self.hidden_previous_timestep = torch.Tensor(self.bs, self.L, self.h).cuda()

        self.RNN_list=nn.ModuleList()
        for _ in range(self.L):
            self.RNN_list.append(LSTM_Unit(self.x, self.R, self.W, self.h, self.bs))

    def forward(self, input):

        hidden_previous_layer = Variable(torch.Tensor(self.bs, self.h).zero_().cuda())
        hidden_this_timestep = Variable(torch.Tensor(self.bs, self.L, self.h).cuda())

        for i in range(self.L):
            hidden_output = self.RNN_list[i](input, self.hidden_previous_timestep[:, i, :],
                                             hidden_previous_layer)
            if (hidden_output!=hidden_output).any():
                raise ValueError("We have NAN in controller output.")
            hidden_this_timestep[:, i, :] = hidden_output
            hidden_previous_layer = hidden_output


        flat_hidden = hidden_this_timestep.view((self.bs, self.L * self.h))
        output = torch.matmul(flat_hidden, self.W_y)
        interface_input = torch.matmul(flat_hidden, self.W_E)
        self.hidden_previous_timestep = hidden_this_timestep
        return output


# lstm=LayeredLSTM()
lstm=LSTM(47764, 128)

# this has no new sequence reset
# I wonder if gradient information will increase indefinitely

# Even so, I think detaching at the beginning of each new sequence is an arbitrary decision.
optim=torch.optim.Adam(lstm.parameters())
lstm.cuda()
states=None
input=None
target=None
output=None


for _ in range(1000):
    print(_)
    optim.zero_grad()
    input=Variable(torch.rand(128,1,47764)).cuda()
    target=Variable(torch.rand(128,1,128)).cuda()
    output, states=lstm(input, states)
    criterion=torch.nn.SmoothL1Loss()
    loss=criterion(output,target)
    loss.backward()
    optim.step()