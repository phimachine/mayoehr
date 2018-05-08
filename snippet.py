import torch
#
# unsorted=torch.Tensor([[1,2,3,4],[9,8,7,6]])
# print(unsorted)
# sorted,indices=unsorted.sort(dim=1)
# print(sorted)
# print(indices)
# ret=torch.gather(sorted,1,indices)
# print(ret)

a=torch.Tensor([[1,2,3],[2,3,4]])
b=torch.Tensor([[2,3]])
print(a)
print(b)
print(a*b.t())