import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class GammaLoss(nn.Module):
    """
    Gamma Loss balances the sensitivity and specificity.
    This is important because our dataset is highly sparse. We aim to predict events, but those events are usually rare.
    The sensitivity and specificity tradeoff can also be addressed by changing the death_fold parameter in InputGen.
    InputGen modification is not finalized, but should be preferred, because it is a more effcient way to compute data.
    """
    def __init__(self,gamma):
        self.gamma=gamma
        #TODO

    def forward(self,loss,loss_type):
        pass

class TOELoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(TOELoss, self).__init__()
        self.real_criterion= nn.SmoothL1Loss(reduce=False)
        self.size_average=size_average
        self.reduce=reduce

    def forward(self, input, target, loss_type):
        '''
        prefer functions to control statements
        :param input:
        :param target:
        :param loss_type: Whether the record was in death or not. Zero if in, one if not.
                          This is a flag for control offsets, the inverse of in_death
        :return:
        '''

        # for toe, all targets are postiive values
        # positive for overestimate
        # negative for underestimate
        diff=input-target
        # instead of passing input and target to criterion directly, we get the difference and put it against zero
        zeros=torch.zeros_like(target)
        base_loss=self.real_criterion(diff,zeros)

        # offset if not in death record (ones) and positively overestimate

        # only take the positive part
        offset=F.relu(diff)
        offset=self.real_criterion(offset,zeros)
        offset=offset*loss_type

        loss=base_loss-offset

        if not self.reduce:
            return loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def test_toe_loss():
    input=Variable(torch.Tensor([0,1,2,3,4,5,6,7,8,9]))
    target=Variable(torch.Tensor([0,2,0,2,0,-2,8,-2,8,0]))
    loss_type=Variable(torch.LongTensor(10).random_(0,2))
    loss_type=loss_type.float()
    Toe=TOELoss()
    print(input)
    print(target)
    print(loss_type)
    print(Toe(input,target,loss_type))