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



# sparse_write_weighting(torch.rand(6,5),3)

class itertest():
    def __init__(self):
        self.a=[1,2,3,4]
        self.ptr=0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr<4:
            self.ptr+=1
            return self.a[self.ptr]
        else:
            raise StopIteration()

# tt=itertest()
# for i in tt:
#     print (i)

def getval():
    i=0
    while i < 10:
        yield i
        i += 1

for i in getval():
    print(i)

class iterofiter():
    def __init__(self):
        self.a=getval()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.a)
        except StopIteration:
            raise StopIteration()

class iii():
    def __init__(self):
        self.a=10
        self.i=0
    def __iter__(self):
        return self

    def __next__(self):
        if self.i>self.a:
            raise StopIteration()
        else:
            self.i+=1
            return self.i-1

# for i in iii():
#     print(i)
#
a=iterofiter()
for i in a:
    print(i)
# a=iterofiter()
# for i in range(11):
#     print(next(a))
