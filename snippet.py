import torch

x= torch.randn(3,4)
sorted, indices=torch.sort(x,0)

print(x)
print(sorted)
print(indices)
print("second run:")

sorted, indices=torch.sort(x,1)
print(x)
print(sorted)
print(indices)

print("integer run")
x=torch.Tensor([[1,2,4,5],[61,4,2,9],[0,-1,8,-3]])
sorted, indices=torch.sort(x,0)

print(x)
print(sorted)
print(indices)
print("second run:")

sorted, indices=torch.sort(x,1)
print(x)
print(sorted)
print(indices)