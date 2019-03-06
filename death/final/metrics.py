# The metrics are extremely important. I'm relying on them more than the loss function to evaluate the models.
# They have to be right and behave as expected.


import torch
import numpy as np
from torch.autograd import Variable

class Evaluate():
    def __init__(self,memory_len=50):
        pass


    def get_stats(self,output,target):
        batch_sensitivity, truepositive, condition_positive = sensitivity(output, target)
        batch_specificity, truenegative, condition_negative = specificity(output, target)

        positives= output





def sensitivity(output, target):
    # TODO I do not think these formula are correct either.
    # Confusion matrix always lose information if projected to these statistics
    '''
    Because batches have varied number of positive labels, we must keep that information for weighted averages.

    :param target: np array of bool
    :param output: np array of float
    :return:
    batch sensitivity
    true positive and positive are needed, because not all batches have the same positive counts
    '''

    # target is one

    truepositive=target*output
    condition_positive=target
    condition_positive=condition_positive.clamp(min=1e-8)

    sensitivity_for_each_class=truepositive/condition_positive
    # across batch dimension
    batch_sensitivity=torch.mean(sensitivity_for_each_class,dim=0)

    return batch_sensitivity.data[0], truepositive.data.numpy(), condition_positive.data.numpy()

def specificity(output, target):
    truenegative=(1-target)*(1-output)
    condition_negative=1-target
    condition_negative=condition_negative.clamp(min=1e-8)

    specificity_for_each_class=truenegative/condition_negative
    batch_specificity=torch.mean(specificity_for_each_class)

    return batch_specificity.data[0], truenegative.data.numpy(), condition_negative.data.numpy()


def precision(output, target):
    truepositive = output * target
    positives = output

    batch_precision = torch.mean(truepositive / positives)

    return batch_precision

def recall(output, target):
    return sensitivity(output, target)


def f1score(output, target):
    # lol where did I get this formula before?
    rec=recall(output,target)
    if rec<1e-6:
        rec=1e-6
    prec=precision(output,target)
    if prec<1e-6:
        prec=1e-6
    f1=1/((1/rec+1/prec)/2)
    return f1

def accuracy(output,target):
    truepositive=torch.sum(output*target).data[0]
    truenegative=torch.sum((1-target)*(1-output)).data[0]
    inc=target.nelement()

    return (truenegative+truepositive)/inc

def smalltest():
    target=Variable(torch.Tensor([1,1,1,1,0,0]))
    output=Variable(torch.Tensor([0,1,1,1,1,0]))
    # sensitivity: 75%
    # specificity: 50%
    # precision: 75%

    print(sensitivity(target,output))
    print(specificity(target,output))
    print(precision(target,output))


if __name__=="__main__":
    smalltest()