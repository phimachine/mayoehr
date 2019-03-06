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

        # for toe, all targets are positive values
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

class WeightedBCELLoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss with positive weights
    Custom made for PyTorch 0.3.1
    Most parameters are ignored to be default
    """
    def __init__(self,pos_weight=None):
        super(WeightedBCELLoss, self).__init__()
        self.pos_weight=pos_weight

    def forward(self, input, target):
        input=torch.clamp(input,min=1e-8,max=1-1e-8)
        info=F.logsigmoid(input)
        if self.pos_weight is not None:
            pos_loss=info*target*self.pos_weight
        else:
            pos_loss=info*target

        neginfo=torch.log(1-F.sigmoid(input))
        neg_loss=neginfo*(1-target)

        return torch.mean(-(pos_loss+neg_loss))

class MyBCEWithLogitsLoss(nn.Module):
    # this loss does not go down at optimization stage.
    # it flucutates slightly, which suggests the variance among the inputs
    # however, the backwards function does not seem to work?
    # this is the problem with positive weights too.

    def __init__(self):
        super(MyBCEWithLogitsLoss, self).__init__()

    def forward(self, input, target):
        # I am using this function because of the numeric issue.
        input=torch.clamp(input,min=1e-8,max=1-1e-8)
        info=F.logsigmoid(input)
        pos_loss=info*target

        neginfo=torch.log(1-F.sigmoid(input))
        neg_loss=neginfo*(1-target)

        return torch.mean(-(pos_loss+neg_loss))

class AnotherBCEWithLogits(nn.Module):
    def __init__(self):
        super(AnotherBCEWithLogits, self).__init__()
        self.bce=nn.BCELoss()

    def forward(self,  input, target):
        # the inputs are logits, clamped to safety
        input=torch.clamp(input,min=1e-8,max=1-1e-8)
        prob=F.sigmoid(input)

        loss=self.bce(prob,target)

        return loss


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

def test_two_losses():
    # what is the difference between the two losses? The problem is that backwards does not seem to go through
    # I tested this before. Forwards are fine, but this time, the focus is on the backwards function.

    torchBCEWLL=nn.BCEWithLogitsLoss()
    myBCEWLL=MyBCEWithLogitsLoss()

    torchBCE=nn.BCELoss()
    myBCE=AnotherBCEWithLogits()

    input=Variable(torch.Tensor(1000).uniform_())
    input2=input.clone()
    input3=input.clone()
    input4=input.clone()
    target=Variable(torch.Tensor(1000).uniform_())

    input.requires_grad=True
    input2.requires_grad=True
    input3.requires_grad=True
    input4.requires_grad=True

    l1=torchBCEWLL(input,target)
    l2=myBCEWLL(input2,target)

    logits=torch.sigmoid(input3)

    l3=torchBCE(logits,target)
    l4=myBCE(input4,target)

    print("torch bcewll\n",l1)
    print("my bcewll\n",l2)

    print("torch bce\n",l3)
    print("my bce\n",l4)

    l1.backward()
    l2.backward()
    l3.backward()
    l4.backward()

    print(input.grad)
    print(input2.grad)
    print(input3.grad)
    print(input4.grad)

    # all gradients are the same. wow
    # okay maybe it's not the problem.
    # maybe the problem is the optimizer cannot deal with another loss.

if __name__ == '__main__':
    test_two_losses()