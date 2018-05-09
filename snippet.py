from archi.computer import Computer
import torch
from pathlib import Path
import os
from torch.optim import SGD
from os.path import abspath

def save_model(net, optim, epoch):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    task_dir = os.path.dirname(abspath(__file__))
    print(task_dir)
    pickle_file=Path("saves/DNC_"+str(epoch)+".pkl")

    torch.save({
        'epoch': 10,
        'state_dict': state_dict,
        'optimizer': optim},
        pickle_file)

computer=Computer()
optim=SGD(computer.parameters(),lr=0.1)

save_model(computer,optim,10)