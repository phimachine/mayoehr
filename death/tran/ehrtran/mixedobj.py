# https://www.aaai.org/Papers/AAAI/2019/AAAI-SachanD.7236.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from death.tran.ehrtranforward.layers import *


class AT(nn.Module):
    def __init__(self, after_embedding, epsilon=0.05):
        """
        AT adds a normalized vector upon the input embedding that is in the direction that increases loss by \epsilon
        The model is required to perform as well with the adversarial data point.

        Because the partial derivative to the loss is calculated, a pass through the original model needs to be run first

        AT does not include the CE criterion.

        embedding is of shape (batch, embedding)

        AT is generated from embedding_grad, which requires targets.
        """
        super(AT, self).__init__()
        # the after_embedding model
        # the original, not the copy. object, not a class
        self.after_embedding = after_embedding
        self.epsilon = epsilon

    def forward(self, embedding, embedding_grad):
        """

        :param embedding:
        :return: free form R
        """
        # embedding=embedding.detach()
        # embedding.requires_grad=True
        # embedding.retain_grad()
        radv = self.epsilon * embedding_grad * norm2reci(embedding_grad)

        new_embed = embedding + radv
        # new input half way in the model.
        output = self.after_embedding(new_embed)
        return output


class EM(nn.Module):
    def __init__(self):
        super(EM, self).__init__()

    def forward(self, output):
        # the output is logit
        prob = F.softmax(output, dim=1)
        logprob = F.log_softmax(output, dim=1)
        batch_loss = -torch.sum(prob * logprob, dim=1)
        loss = batch_loss.mean(dim=0)
        return loss


class VAT(nn.Module):
    """
    VAT is generated from embedding itself and not embedding, hence unsupervised.
    """

    def __init__(self, after_embedding, xi=1):
        super(VAT, self).__init__()
        self.after_embedding = after_embedding
        self.xi = xi

    def get_g(self, embedding, output):
        """
        :param embedding:
        :param output:
        :return:
        """
        # detach, because g is a noise generator, does not require grad.
        # detach returns a new tensor and does not require grad, so we make it.
        embedding = embedding.detach()
        # embedding.requires_grad=True
        # embedding.retain_grad()
        noise_sample = torch.zeros_like(embedding).normal_()
        xid = self.xi * noise_sample * norm2reci(noise_sample)
        vprimei = embedding + xid
        # none of this requires grad so far
        afterpipe = self.after_embedding(vprimei)
        beforepipe = output
        # KL divergence
        dkl = self.kl_divergence(beforepipe, afterpipe).sum()
        # this will be backed to embedding? to avoid interference, I should detach embedding
        embedding_grad = torch.autograd.grad(dkl, embedding, retain_graph=True, only_inputs=True)[0]
        return embedding_grad

    def kl_divergence(self, beforepipe, afterpipe):
        """

        :param beforepipe: free form R
        :param afterpipe: ffR, same dimension
        :return: dkl: unsumed, dimension equal to either pipe
        """
        p = F.softmax(beforepipe, dim=1)
        beforepipe = F.log_softmax(beforepipe, dim=1)
        afterpipe = F.log_softmax(afterpipe, dim=1)
        dkl = p * (beforepipe - afterpipe)
        return dkl

    def forward(self, out, vout):
        """

        :param vout: ffR, prediction
        :param out: ffR, prediction
        :return:
        """
        dkl = self.kl_divergence(out, vout).sum(dim=1).mean(dim=0)
        return dkl


if __name__ == '__main__':
    pass