from death.taco.model import *




class SmallDecoder(nn.Module):
    """
    Decoder
    """
    def __init__(self):
        super(SmallDecoder, self).__init__()
        self.prenet = SecondPrenet(hp.decoder_output_dim, hp.hidden_size * 2, hp.hidden_size)
        self.attn_decoder = AttentionDecoder(hp.target_size)

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

        # [batch, hidden*2, time*timecoef]
        return outputs


class SmallTaco(nn.Module):
    """
    End-to-end Tacotron Network
    """
    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.input_size,hp.embedding_size)
        self.decoder = SmallDecoder()

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
    taco=Tacotron().cuda()
    output=taco(input)
    print("script finished")
    print(output)


if __name__=="__main__":
    main()