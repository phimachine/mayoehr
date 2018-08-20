# RNN reads a whole sequence, and then outputs the target sequences at <GO> frame.
# If you think about it this way, can't we read the whole sequence and output the goal at <GO> frame just once?
# Our goal is to produce a target sequence of length one.
# In this case, would attention be necessary? Wouldn't attention be a vanilla linear module?

# one question left: how to deal with variable input lengths?

# I'm still unsure how to use attention. The deadline is compelling, and I think I will not incorporate attention
# mechanism in the architecture.
# A simple way will be a fixed iteration attention RNN wherein the last hidden will be used to compute the target.


"""
The model architecture will be almost exactly the tacotron original architecture.
The only difference is that after the second CBHG, the GL recons is replaced with a maxpool to target.

This design choice is made for these two reasons:
I want to retain the attention mechanism to allow the computer to calculate the target with a few steps, which simulate
a tree like reasoning process. If not, I will only use CBHG to do a multi window convolution on patient history, which
is not very interesting/powerful.
I want to retain the last CBHG module, because if I directly maxpool the last hidden without processing it, the hidden
will need to be on the target space and annotation space at the same time. It's not impossible, given that annotation
is on the space of input hidden, which is a medical record, while the target is also a medical record. But I cannot
be 100% sure the control will not interfere with the target, so I will allow CBHG. CBHG can trivially emulate an
identity transformation, so if maxpool directly is preferred, then this model should not introduce any further bias.
Variance certainly increases.
"""


import torch.nn as nn
import torch
from death.taco.module import *


class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, embedding_size):
        """

        :param embedding_size: dimension of embedding
        """
        super(Encoder, self).__init__()
        # we do not need an embedding, because our input is not one hot, but multihot.
        # replace it with a linear
        # self.embedding_size = embedding_size
        # self.embed = nn.Embedding(len(symbols), embedding_size)

        self.lin = nn.Linear(self.x, embedding_size)
        self.prenet = Prenet(embedding_size, hp.hidden_size * 2, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)

    def forward(self, input_):

        input_ = torch.transpose(self.lin(input_),1,2)
        prenet = self.prenet(input_)
        memory = self.cbhg(prenet)

        return memory

class Tacotron(nn.Module):
    """
    End-to-end Tacotron Network
    """
    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.embedding_size)
        self.decoder1 = MelDecoder()
        self.decoder2 = PostProcessingNet()

    def forward(self, characters, mel_input):
        memory = self.encoder(characters)
        mel_output = self.decoder1(mel_input, memory)
        linear_output = self.decoder2(mel_output)

        return mel_output, linear_output

