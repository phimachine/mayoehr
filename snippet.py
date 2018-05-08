import torch

unsorted=torch.Tensor([[1,2,3,4],[9,8,7,6]])
print(unsorted)
sorted,indices=unsorted.sort(dim=1)
print(sorted)
print(indices)
ret=torch.gather(sorted,1,indices)
print(ret)