from death.tran.ehrtran.layers import *
from death.tran.ehrtran.helpers import *

class TransformerMixedForward(nn.Module):
    # no attention module here, because the input is not timewise
    # we will add time-wise attention to time-series model later.

    def __init__(self, prior, binary_criterion, real_criterion, d_model=256, input_size=50000, d_inner=32, dropout=0.1,
                 n_layers=8, output_size=12, epsilon=1, xi=1, beta=0.01,
                 lambda_ml=1, lambda_at=1, lambda_em=1, lambda_vat=1):
        super(TransformerMixedForward, self).__init__()

        self.d_inner = d_inner
        self.d_model = d_model
        self.n_layer = n_layers
        self.dropout = dropout
        self.input_size = input_size
        self.embedding = torch.nn.Parameter(torch.Tensor(input_size, d_model))
        # self.first_embedding=nn.Linear(vocab_size,d_model)
        self.layer_stack = nn.ModuleList([
            EncoderLayerForward(d_model, d_inner, dropout=dropout)
            for _ in range(n_layers)])
        self.last_linear = nn.Linear(d_model, output_size)
        # always holds reference to a set of embedding
        # should not be a memory issue, since it's released after every training set
        # however, after training finished, unless the model is released, this tensor will remain on the GPU
        self.epsilon = epsilon

        self.lambda_ml = lambda_ml
        self.lambda_at = lambda_at
        self.lambda_em = lambda_em
        self.lambda_vat = lambda_vat
        self.xi = xi
        self.beta=beta

        self.binary_criterion = binary_criterion
        self.real_criterion = real_criterion

        '''prior'''
        # this is the prior probability of each label predicting true
        # this is added to the logit
        if prior is not None:

            self.prior=prior
            if isinstance(self.prior, np.ndarray):
                self.prior=torch.from_numpy(self.prior).float()
                self.prior=Variable(self.prior, requires_grad=False)
            elif isinstance(self.prior, torch.Tensor):
                self.prior=Variable(self.prior, requires_grad=False)
            else:
                assert(isinstance(self.prior, Variable))

            # transform to logits
            # because we are using sigmoid, not softmax, self.prior=log(P(y))-log(P(not y))
            # sigmoid_input = z + self.prior
            # z = log(P(x|y)) - log(P(x|not y))
            # sigmoid output is the posterior positive
            self.prior=self.prior.clamp(1e-8, 1 - 1e-8)
            self.prior=torch.log(self.prior)-torch.log(1-self.prior)
            a=Variable(torch.Tensor([0]))
            self.prior=torch.cat((a,self.prior))
            self.prior=self.prior.cuda()

        else:
            self.prior=None

        self.reset_parameters()
    @staticmethod
    def reset_mod(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.embedding.data)
        self.apply(self.reset_mod)

    def one_pass(self, input, input_length, target, loss_type):

        # none of these functions work for binary classifications. Rework them.
        # ml
        output = self(input, input_length)
        toe_output = output[:, 0]
        time_loss = self.real_criterion(toe_output, target[:, 0], loss_type)
        cod_output = output[:, 1:]
        cod_target = target[:, 1:]
        lml = self.binary_criterion(cod_output, cod_target)

        # adv, at
        embed_grad = torch.autograd.grad(lml, self.embedding, only_inputs=True, retain_graph=True)[0]
        radv = self.radv(embed_grad)
        yat = self(input, input_length, radv)
        lat = self.binary_criterion(yat[:, 1:], cod_target)

        # unsupervised
        lem = self.em(cod_output)

        # vat
        xid = self.xid()
        aoutput = self(input, input_length, xid)
        rvat = self.rvat(cod_output, aoutput[:, 1:])
        yvat = self(input, input_length, rvat)
        lvat = self.binary_kl_divergence(cod_output, yvat[:, 1:])
        lvat = lvat.mean(dim=1).mean(dim=0)

        all_loss = self.lambda_ml * lml + self.lambda_at * lat + self.lambda_em * lem + self.lambda_at * lvat \
                   + self.beta * time_loss
        return all_loss, lml, lat, lem, lvat, output

    def forward(self, input, input_length, r=None):
        """
        pass one time with embedding_grad=None

        :param input: (batch_size, embedding)
        :param input_length: ignored
        :param r: for mixed objective loss
        :return:
        """

        input=input.max(dim=1)[0]
        if r is None:
            enc_output = torch.matmul(input, self.embedding)
        else:
            enc_output = torch.matmul(input, (self.embedding + r))

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        output = self.last_linear(enc_output)
        output = output.squeeze(1)
        if self.prior is not None:
            return output + self.prior
        else:
            return output

    def radv(self, embedding_grad):
        """
        computatino of embedding_grad should not modify the graph and the graph's grad in any way.
        pass this to the forward
        """
        radv = self.epsilon * embedding_grad * norm2reci(embedding_grad)
        return radv

    def xid(self):
        """ pass this to the forward then KLD """
        noise_sample = torch.zeros_like(self.embedding).normal_()
        xid = self.xi * noise_sample * norm2reci(noise_sample)
        return xid

    def rvat(self, output, aoutput):
        """
        this function should not incur any gradient backprop
        xid is sampled from the thin-air, passed to forward argument, so the whole model is included in the graph
        to compute dkl and g. However, the backprop is configured to not modify any gradient of the models, certainly
        not values. So when autograd takes care of its business, nothing here will be considered. To autograd,
        rvat should be just a tensor.
        Despite previous gradient passes, g should not be influenced, because derivative, well, is derivative.

        :param output: free form R
        :param aoutput: adversarial output, free form R, returned by passing xid to forward
        :return:
        """
        # embed_copy=self.embedding.detach()
        beforepipe = output
        afterpipe = aoutput
        dkl = self.binary_kl_divergence(beforepipe, afterpipe)
        # this pass does not accumulate the gradient. since this is the only backward pass
        dkl = dkl.sum(dim=1).mean(dim=0)
        g = torch.autograd.grad(dkl, self.embedding, retain_graph=True, only_inputs=True)[0]
        rvat = self.radv(g)
        return rvat

    def em(self,output):
        """
        treating this function as a functional should work? I hope? I do not know for sure.
        if does not work, wrap it with nn.Module as a forward()
        :param output:
        :return:
        """
        # the output is logit
        loss=self.binary_criterion(output,output)
        return loss

    @staticmethod
    def kl_divergence(beforepipe, afterpipe):
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

    @staticmethod
    def binary_kl_divergence(beforepipe,afterpipe):
        """
        An alterantive kl_divergence is provided here for binary labels.
        Note that this is an approximation of the joint kl divergence.
        :param beforepipe:
        :param afterpipe:
        :return:
        """
        beforep=torch.sigmoid(beforepipe)
        afterp=torch.sigmoid(afterpipe)
        beforep=beforep.clamp(1e-6, 1-1e-6)
        afterp=afterp.clamp(1e-6, 1-1e-6)
        pos=beforepipe*torch.log(beforep/afterp)
        neg=(1-beforep)*torch.log((1-beforep)/(1-afterp))
        return pos+neg

if __name__ == '__main__':
    from death.post.inputgen_planJ import InputGenJ, pad_collate
    from death.final.losses import TOELoss
    real_criterion = TOELoss()
    binary_criterion= nn.BCEWithLogitsLoss()
    igj=InputGenJ()
    trainig=igj.get_train()
    d1=trainig[10]
    d2=trainig[11]
    # input=torch.empty(64,400,7298).uniform_()
    # target=torch.empty(64,435).uniform_()
    input, target, loss_type, time_length=[t.cuda() for t in pad_collate((d1,d2))]
    model=TransformerMixedForward(binary_criterion=binary_criterion,real_criterion=real_criterion,
                               input_size=7298, output_size=435, prior=None).cuda()
    all_loss, lml, lat, lem, lvat, output=model.one_pass(input, time_length, target, loss_type)
    print(all_loss, lml, lat, lem, lvat, output)
