### The problem
I am assuming that all three models are stuck at some modes that are
not dependent on paricular inputs. That is, model outputs 12% for all
classification tasks, because it's the background probability.
Relationships between input and output is not captured.

This assumption is not evidenced.

If this assumption is true, it means, for the input-target relationship,
the signal failed to transmit. That is, for large sample, the derivative
converges to a direction that is not aligned with this relationship.

Why is the signal so weak? Does the relationship have huge noise?

### The story

The dataset is very difficult to work with. Imagine for sake of
thought experiement that this dataset and an image dataset were
somehow was leaked with no labels on them, together with the structure
of the model.

Now the hacker needs to guess what the datasets mean, through revsere
engineering.

With image dataset, the hacker knows it's image very easily, by looking
at the convolution structure. It knows how many channels, pixels.
However, it's not easy to understand what this model captures, if
without any prior from human experience. That is, you can test the
model with a bunch of ocmmon pictures and guess what it means, but
you need to know what are "common picutres" to sample from, which
requires a cultural prior not from data. If without such cultural
prior, you would not know what the model captures. That's why such
models are hackable.

With our dataset, the hacker knows it's a time-dependent series
that is largely sparse.
The hacker knows it's a bunch of events, but what event?
The hacker has no clue whatsoever. For each step, the input is fed
in as a flattened vector, and the output is a flattened vector.
There is no structure that is told from the model or anything.

This is a very bad design from our dataset and our model. An image or a
wave can be both flattened to be a vector and directly sent into the
model, but we don't do that.
That's why neural network failed in the early days.
The power of LeCun's convolution net is because of,
well, the convolution.
It bakes the property of image in the network.

What is our property? What bakes our property in the network?

We need structured codes. But we should not send it in as flattened
codes. What model can effectively process trees? How does DNC
deal with trees?

If the input is a tree structure, shouldn't the processing pipeline
be a tree too? The deeper nodes do not have enough training points, but
naturally that should be the case. This means the end nodes should
refrain from influcing in the final output by default. Identity
transform or gate. Does skip connection help?