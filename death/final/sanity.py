import torch
import torch.nn as nn

class Sanity():

    def __init__(self,batch_size,output_dim):
        self.output_dim=output_dim
        self.batch_size=batch_size

    def forward(self, input):
        return torch.zeros((self.batch_size, self.output_dim))