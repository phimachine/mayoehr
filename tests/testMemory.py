import torch
from archi.memory import Memory
import archi.param as param
import unittest
from torch.autograd import Variable

class TestMemory(unittest.TestCase):

    def test_update_temporal_linkage_matrix(self):
        memory=Memory()
        temp=param.N
        param.N=3
        write_weighting = torch.Tensor([1, 2, 3])
        memory.precedence_weighting = torch.Tensor([5, 6, 7])
        memory.temporal_memory_linkage = torch.Tensor(param.N, param.N).fill_(1)
        val = memory.update_temporal_linkage_matrix(write_weighting)
        self.assertTrue(val.equal(torch.Tensor([[4,4,4],[8,9,10],[12,14,16]])))
        param.N=temp

    def test_memory_allocation_weighting(self):
        memory=Memory()
        val=memory.allocation_weighting(torch.Tensor([2,-5,2,2]))
        self.assertTrue(val.equal(torch.Tensor([5,6,10,20])))

    def test_write_content_weighting(self):
        temp_param_N = param.N
        param.N = 2
        temp_param_W = param.W
        param.W = 3
        memory = Memory()
        memory.memory=Variable(torch.Tensor([[1,1,1],[-1,-1,-1]]))
        val=memory.write_content_weighting(Variable(torch.Tensor([1,1,1])),1)
        self.assertTrue(val.equal(torch.nn.functional.softmax(Variable(torch.Tensor([1,-1])),dim=0)))
        param.N=temp_param_N
        param.W=temp_param_W

    def test_read_content_weighting(self):
        temp_param_N=param.N
        param.N=2
        temp_param_W=param.W
        param.W=3
        memory=Memory()
        memory.memory=Variable(torch.Tensor([[1,1,1],[-1,-1,-1]]))
        print(memory.read_content_weighting(Variable(torch.Tensor([[1,1,1],[2,3,4]])),1))
        param.N=temp_param_N
        param.W=temp_param_W
