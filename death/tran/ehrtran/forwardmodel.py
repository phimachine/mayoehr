from death.tran.ehrtran.layers import *
from death.tran.ehrtran.helpers import *

class TransformerMixedForward(nn.Module):
    # no attention module here, because the input is not timewise
    # we will add time-wise attention to time-series model later.

    def __init__(self, prior, d_model=256, input_size=50000, d_inner=32, dropout=0.1, n_layers=8, output_size=12, epsilon=1,
                 xi=1,
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

        '''prior'''
        # this is the prior probability of each label predicting true
        # this is added to the logit
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

        self.reset_parameters()

    @staticmethod
    def reset_mod(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.embedding.data)
        self.apply(self.reset_mod)

    def one_pass(self, input, target):

        # none of these functions work for binary classifications. Rework them.
        # ml
        output = self(input)
        lml = F.cross_entropy(output, target)

        # adv, at
        embed_grad = torch.autograd.grad(lml, self.embedding, only_inputs=True, retain_graph=True)[0]
        radv = self.radv(embed_grad)
        yat = self(input, radv)
        lat = F.cross_entropy(yat, target)

        # unsupervised
        lem = self.em(output)

        # vat
        xid = self.xid()
        aoutput = self(input, xid)
        rvat = self.rvat(output, aoutput)
        yvat = self(input, rvat)
        lvat = self.kl_divergence(output, yvat)
        lvat = lvat.sum(dim=1).mean(dim=0)

        all_loss = self.lambda_ml * lml + self.lambda_at * lat + self.lambda_em * lem + self.lambda_at * lvat
        return all_loss, lml, lat, lem, lvat, output

    def forward(self, input, r=None):
        """
        pass one time with embedding_grad=None

        :param input: (batch_size, embedding)
        :return:
        """
        # enc_output=self.first_embedding(input)
        if r is None:
            enc_output = torch.matmul(input, self.embedding)
        else:
            enc_output = torch.matmul(input, (self.embedding + r))

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        output = self.last_linear(enc_output)
        output = output.squeeze(1)
        return output + self.prior

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
        dkl = self.kl_divergence(beforepipe, afterpipe)
        # this pass does not accumulate the gradient. since this is the only backward pass
        dkl = dkl.sum(dim=1).mean(dim=0)
        g = torch.autograd.grad(dkl, self.embedding, retain_graph=True, only_inputs=True)[0]
        rvat = self.radv(g)
        return rvat

    @staticmethod
    def em(output):
        """
        treating this function as a functional should work? I hope? I do not know for sure.
        if does not work, wrap it with nn.Module as a forward()
        :param output:
        :return:
        """
        # the output is logit
        prob = F.softmax(output, dim=1)
        logprob = F.log_softmax(output, dim=1)
        batch_loss = -torch.sum(prob * logprob, dim=1)
        loss = batch_loss.mean(dim=0)
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
