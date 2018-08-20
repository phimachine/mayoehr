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
from death.taco.attention import *
import random
import death.taco.hyper as hp

class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, input_size, embedding_size):
        """

        :param embedding_size: dimension of embedding
        """
        super(Encoder, self).__init__()
        # no embeddings
        self.lin=nn.Linear(input_size,embedding_size)
        self.prenet = Prenet(embedding_size, hp.hidden_size * 2, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size, projection_size=256)

    def forward(self, input_):

        input_ = torch.transpose(self.lin(input_),1,2)
        # starting from here, the time dim is 2
        prenet = self.prenet(input_)
        memory = self.cbhg(prenet)

        return memory

class MelDecoder(nn.Module):
    """
    Decoder
    """
    def __init__(self):
        super(MelDecoder, self).__init__()
        self.prenet = Prenet(hp.num_mels, hp.hidden_size * 2, hp.hidden_size)
        self.attn_decoder = AttentionDecoder(hp.hidden_size * 2)

    def forward(self, decoder_input, memory):

        # Initialize hidden state of GRUcells
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(decoder_input.size()[0])
        outputs = list()

        # Training phase
        if self.training:
            # Prenet
            dec_input = self.prenet(decoder_input)
            timesteps = dec_input.size()[2] // hp.outputs_per_step

            # [GO] Frame
            prev_output = dec_input[:, :, 0]

            for i in range(timesteps):
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder(prev_output, memory,
                                                                                             attn_hidden=attn_hidden,
                                                                                             gru1_hidden=gru1_hidden,
                                                                                             gru2_hidden=gru2_hidden)

                outputs.append(prev_output)

                if random.random() < hp.teacher_forcing_ratio:
                    # Get spectrum at rth position
                    prev_output = dec_input[:, :, i * hp.outputs_per_step]
                else:
                    # Get last output
                    prev_output = prev_output[:, :, -1]

            # Concatenate all mel spectrogram
            outputs = torch.cat(outputs, 2)

        else:
            # [GO] Frame
            prev_output = decoder_input

            for i in range(hp.max_iters):
                prev_output = self.prenet(prev_output)
                prev_output = prev_output[:,:,0]
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder(prev_output, memory,
                                                                                         attn_hidden=attn_hidden,
                                                                                         gru1_hidden=gru1_hidden,
                                                                                         gru2_hidden=gru2_hidden)
                outputs.append(prev_output)
                prev_output = prev_output[:, :, -1].unsqueeze(2)

            outputs = torch.cat(outputs, 2)

        return outputs

class PostProcessingNet(nn.Module):
    """
    Post-processing Network
    """
    def __init__(self):
        super(PostProcessingNet, self).__init__()
        self.postcbhg = CBHG(hp.hidden_size,
                             K=8,
                             projection_size=hp.num_mels,
                             is_post=True)
        self.linear = SeqLinear(hp.hidden_size * 2,
                                hp.num_freq)

    def forward(self, input_):
        out = self.postcbhg(input_)
        out = self.linear(torch.transpose(out,1,2))

        return out

class Tacotron(nn.Module):
    """
    End-to-end Tacotron Network
    """
    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.input_size,hp.embedding_size)
        self.decoder1 = MelDecoder()
        self.decoder2 = PostProcessingNet()

    def forward(self, characters, mel_input):
        # .forward() should never be explicitly called.
        # memory is just a character feature, with dimension of (batch, time, hidden_size*2)
        memory = self.encoder(characters)
        mel_output = self.decoder1(mel_input, memory)
        linear_output = self.decoder2(mel_output)

        return mel_output, linear_output
def main():
    input=Variable(torch.rand((123,1, 47774))).cuda()
    mel_input=Variable(torch.rand((123,hp.num_mels))).cuda()
    taco=Tacotron().cuda()
    output=taco(input, mel_input)
    print("script finished")


if __name__=="__main__":
    main()