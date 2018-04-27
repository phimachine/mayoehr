import torch
from archi.memory import Memory
import archi.param as param
import unittest
from torch.autograd import Variable

class Test_Memory_Necessary(unittest.TestCase):
    # includes a set of necessary tests
    # there is no way I know beforehand what the values should be
    # step over the debugger and see for yourself
    
    def overwrite_memory(self):
        temp_param_N=param.N
        param.N=2
        temp_param_W=param.W
        param.W=3
        memory=Memory()
        memory.reset_parameters()
        memory.memory=torch.Tensor([[1,1,1],[-1,-1,-1]])
        memory.temporal_memory_linkage=torch.Tensor([[1,7],[7,1]])
        return memory

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
        # two locations, width of three
        param.N=2
        param.W=3
        param.R=2
        memory=Memory()
        memory.memory=torch.Tensor([[1,1,1],[-1,-1,-1]])
        read_keys=torch.Tensor([[1,1,1],[2,3,4]]).t()
        rcw=(memory.read_content_weighting(read_keys,torch.Tensor([0.5,1])))
        self.assertTrue(rcw.data.size()==(param.N, param.R))

    def test_forward_backward_weighting(self):
        memory=self.overwrite_memory()
        bw=memory.backward_weighting()
        fw=memory.forward_weighting()
        self.assertTrue(fw == torch.Tensor([[8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8]]))
        self.assertTrue(bw == torch.Tensor([[8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8]]))

    def test_read_weighting(self):
        memory=self.overwrite_memory()
        fw=memory.forward_weighting()
        bw=memory.backward_weighting()
        read_keys=torch.Tensor(param.W,param.R).normal_()
        read_key_strengths=torch.Tensor(param.R).normal_()
        read_modes=torch.Tensor(param.R,3).normal_()
        rw=memory.read_weighting(fw,bw,read_keys,read_key_strengths,read_modes)

    def test_allocation_gate(self):
        pass

    def test_write_weighting(self):
        memory=self.overwrite_memory()
        write_key=torch.Tensor([1,2,3])
        write_strength=0.5
        allocation_gate=0.56
        write_gate=0.47
        allocation_weighting=memory.write_content_weighting(write_key,write_strength)
        ww=memory.write_weighting(write_key,write_strength,allocation_gate,
                                  write_gate,allocation_weighting)

    def test_write_to_memory_i(self,read_weighting_i):
        pass
