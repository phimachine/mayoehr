from dnc import DNC
import torch
from torch.autograd import Variable

rnn = DNC(
  input_size=64,
  hidden_size=128,
  rnn_type='lstm',
  num_layers=4,
  nr_cells=100,
  cell_size=32,
  read_heads=4,
  batch_first=True,
  gpu_id=0
)

(controller_hidden, memory, read_vectors) = (None, None, None)

for i in rnn.modules():
    i.cuda()

output, (controller_hidden, memory, read_vectors) = \
    rnn(Variable(torch.randn(10, 4, 64)).cuda(), (controller_hidden, memory, read_vectors), reset_experience=True)

print(output)