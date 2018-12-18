from torch.nn.modules import LSTM
import torch

computer=LSTM(100,128)
lr=1e-3
optimizer = torch.optim.Adam(computer.parameters(), lr=lr)
for group in optimizer.param_groups:
    print(group["lr"])