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
        self.prenet = SecondPrenet(hp.decoder_output_dim, hp.hidden_size * 2, hp.hidden_size)
        self.attn_decoder = AttentionDecoder(hp.hidden_size * 2)

    def forward(self, memory):

        # Initialize hidden state of GRUcells
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(memory.size()[0])
        outputs = list()

        # Training phase
        if self.training:
            # # Prenet
            # dec_input = self.prenet(decoder_input)
            timesteps = memory.size()[1] // hp.outputs_per_step
            #
            # # [GO] Frame
            # # [batch, 2*hidden]
            # prev_output = dec_input[:, :, 0]
            prev_output = Variable(torch.zeros(memory.shape[0], hp.decoder_output_dim)).cuda()

            for i in range(timesteps):

                # on the paper, the previous output passes through prenet every loop
                # here it does not.
                # I made a new prenet so that it can be passed through
                prev_output = self.prenet(prev_output)

                # attn_hidden and all gru_hidden are all the same.
                # the RNN takes the prev decoder output, processes it, then directly put it against memory for attention
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder(prev_output, memory,
                                                                                             attn_hidden=attn_hidden,
                                                                                             gru1_hidden=gru1_hidden,
                                                                                             gru2_hidden=gru2_hidden)

                # (batch_num, hidden, outputs_per_step)
                outputs.append(prev_output)
                '''
                # what is this line? default param is 100%
                # as I found out, https://arxiv.org/pdf/1506.03099.pdf
                # this is the curriculum training that switches from a fully guided mel_input to blind backprop
                # we cannot do this, because they have the correct mel_spectrogram. We have nothing
                # this training scheme is very interesting. You train the model pipeline by dividing it
                # to two models, and feed two sets of labels one at the end and one in the middle.
                # very good idea. Encourages convergence on interpretable models.
                # but we cannot use it.
                if random.random() < hp.teacher_forcing_ratio:
                    # Get spectrum at rth position
                    prev_output = dec_input[:, :, i * hp.outputs_per_step]
                else:
                    # Get last output
                    prev_output = prev_output[:, :, -1]
                '''
                # this is the r=3 and taking the last one. as in paper
                prev_output=prev_output[:,:,-1]
            # Concatenate all mel spectrogram
            outputs = torch.cat(outputs, 2)

        else:
            # [GO] Frame
            prev_output = Variable(torch.zeros(memory.shape[0], hp.decoder_output_dim)).cuda()
            #(8, 256)
            for i in range(hp.max_iters):
                prev_output = self.prenet(prev_output)
                # prev_output = prev_output[:,:,0]
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder(prev_output, memory,
                                                                                         attn_hidden=attn_hidden,
                                                                                         gru1_hidden=gru1_hidden,
                                                                                         gru2_hidden=gru2_hidden)
                outputs.append(prev_output)
                prev_output = prev_output[:, :, -1]#.unsqueeze(2)

            outputs = torch.cat(outputs, 2)

        # [batch, hidden*2, time*timecoef]
        return outputs

class PostProcessingNet(nn.Module):
    """
    Post-processing Network
    """
    def __init__(self):
        super(PostProcessingNet, self).__init__()
        self.postcbhg = CBHG(hp.hidden_size,
                             K=8,
                             projection_size=hp.decoder_output_dim,
                             is_post=True)
        # here is a modification.
        # we take max directly
        self.linear = SeqLinear(hp.hidden_size * 2,
                                hp.target_size)

    def forward(self, input_):
        out = self.postcbhg(input_)
        out = self.linear(torch.transpose(out,1,2))
        out, idx = torch.max(out,dim=2)
        return out

class Tacotron(nn.Module):
    """
    End-to-end Tacotron Network
    """
    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.input_size,hp.embedding_size)
        self.decoder = MelDecoder()
        self.postp = PostProcessingNet()

    def forward(self, characters):
        # .forward() should never be explicitly called.
        # memory is just a character feature, with dimension of (batch, time, hidden_size*2)
        memory = self.encoder(characters)
        decoder_output = self.decoder(memory)
        output = self.postp(decoder_output)

        return output

def main():
    # because the algorithm requires time wise convolution as well as a decoder whose length is in
    # proportion to the input, we need to feed the whole time-wise sequence in the machine.
    input=Variable(torch.rand((16, 100, 47774))).cuda()
    mel_input=Variable(torch.rand((16,1, hp.num_mels))).cuda()
    taco=Tacotron().cuda()
    output=taco(input)
    print("script finished")
    print(output)


if __name__=="__main__":
    main()