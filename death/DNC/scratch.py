import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.parameter import Parameter

def sparse_write_weighting(write_weighting, k):
    sorted, indices = torch.sort(write_weighting, 1)
    batch_indices=torch.arange(write_weighting.shape[0],out=torch.LongTensor())
    zero_indices=indices[:,k:]
    # (batch_num,
    batch_indices=batch_indices.unsqueeze(1).expand(write_weighting.shape[0],write_weighting.shape[1]-k)

    zero_indices=zero_indices.contiguous().view(-1).unsqueeze(1)
    batch_indices=batch_indices.contiguous().view(-1).unsqueeze(1)

    indices=torch.cat((zero_indices,batch_indices),1)

    write_weighting[indices] = 0



# sparse_write_weighting(torch.rand(6,5),3)

# xx = Variable(torch.randn(1,1), requires_grad = True)
# yy = 3*xx
# zz = yy**2
# zz.backward()
# print(xx.grad) # This is ok
# print(yy.grad) # This gives 0!
# print(zz.grad) # This should give 1!


# you cannot run this script if there is no parameter in this module, so nothing can be trained.
#
class Inspection(torch.nn.Module):
    def __init__(self):
        super(Inspection, self).__init__()
        # changing the requires_grad flag does not reflect on the variable value.
        self.x=Variable(torch.Tensor([10]),requires_grad=True)
        self.y=Parameter(torch.Tensor([10]))

    def forward(self,input):
        return self.x*self.y*input

mod=Inspection()
optim=torch.optim.Adam([p for p in mod.parameters() if p.requires_grad])
output=mod(Variable(torch.Tensor([10])))
criterion=torch.nn.SmoothL1Loss()
loss=criterion(output,Variable(torch.Tensor([19])))
loss.backward()
optim.step()
print(mod.x)
print(mod.y)