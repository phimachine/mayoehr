import torch
from archi.memory import Memory
import archi.param as param
#
# '''
# Testing sort
# '''
# x= torch.randn(3,4)
# sorted, indices=torch.sort(x,0)
#
# print(x)
# print(sorted)
# print(indices)
# print("second run:")
#
# sorted, indices=torch.sort(x,1)
# print(x)
# print(sorted)
# print(indices)
#
# print("integer run")
# x=torch.Tensor([[1,2,4,5],[61,4,2,9],[0,-1,8,-3]])
# sorted, indices=torch.sort(x,0)
#
# print(x)
# print(sorted)
# print(indices)
# print("second run:")
#
# sorted, indices=torch.sort(x,1)
# print(x)
# print(sorted)
# print(indices)
#
# # not sure if torch.gather() would help.
# print(x.index_select(indices))

#
# '''
# Testing Memory.allocation_weighting()
# '''
# memory= Memory()
# val=memory.allocation_weighting(torch.Tensor([2,-5,2,2]))
# print(val)
#
# '''
# Testing indexing with indices returned by sorting
# '''
# x=torch.Tensor([1,2,3,4,5])
# idx=torch.LongTensor([4,2,3,1,0])
# print(x.index_select(0,idx))
# print(x[idx])
#
#

# '''
# Testing temporal linkage
# '''

# memory=Memory()
# write_weighting=torch.Tensor([1,2,3])
# precedence_weighting=torch.Tensor([5,6,7])
# memory.temporal_memory_linkage=torch.Tensor(param.N, param.N).fill_(1)
# val=memory.update_temporal_linkage_matrix(write_weighting,precedence_weighting)
# print(val)
