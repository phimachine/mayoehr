Model from https://github.com/jadore801120/attention-is-all-you-need-pytorch
Thanks @jadore801120 

The reason why I want to work with Transformer is because the Tacotron model failed
in an unexplanable way. I wnat to know if this is becuase of my modifications of Tacotron,
the domain prior of Tacotron, or if this seq-to-seq model in general does not work.
The reason why I wanted to avoid Transformer in the beginning is because the target
in our dataset is not a sequence. Well. now it's not a drawback anymore, since
Tacotron is aimed to produce sequence too. Tacotron has a narrower application domain,
our experiments suggest that such bias can be a problem, and with a more general model
like Transformer, the bias may be eliminated.