import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numbers

class LayerNorm(nn.Module):
    # have to write this for 0.3.1
    # good thing it's easy, no memorization across data points necessary
    # norm across hiddens, only depend on current time step https://arxiv.org/pdf/1607.06450.pdf

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.fill_(0)

    def forward(self, input):
        # be careful which dimension this is.
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)
        ret = self.weight*(input-mean)/(std+self.eps) + self.bias
        return ret
