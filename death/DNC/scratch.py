import torch
import numpy as np

def sparse_write_weighting(write_weighting, k):
    sorted, indices = torch.sort(write_weighting, 1)
    batch_indices=torch.arange(write_weighting.shape[0],out=torch.LongTensor())
    zero_indices=indices[:,k:]
    # (batch_num,
    batch_indices=batch_indices.unsqueeze(1).expand(write_weighting.shape[0],write_weighting.shape[1]-k)

    zero_indices=zero_indices.contiguous().view(-1).unsqueeze(1)
    batch_indices=batch_indices.contiguous().view(-1).unsqueeze(1)

    indices=torch.cat((zero_indices,batch_indices),1)

    write_weighting[indices] = 0



sparse_write_weighting(torch.rand(6,5),3)