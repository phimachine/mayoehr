import torch
from archi.memory import Memory
import archi.param as param
import unittest
from torch.autograd import Variable
from archi.interface import Interface
from archi.controller import Controller
import pdb


class Test_Memory_Necessary(unittest.TestCase):
    # Note that only memory should be retained with each new sequence.
    # Everything else resets at t=0
    # A question is that should read weights be used this timestep or next timestep.
    
    def overwrite_memory(self):
        memory=Memory()
        memory.reset_parameters()
        return memory

    def test_update_temporal_linkage_matrix(self):
        memory=self.overwrite_memory()
        write_weighting = torch.Tensor(param.bs, param.N)
        val = memory.update_temporal_linkage_matrix(write_weighting)
        self.assertTrue(val.size()==(param.bs, param.N, param.N))

    def test_write_content_weighting(self):
        memory = self.overwrite_memory()
        val=memory.write_content_weighting(torch.Tensor(param.bs,param.W),torch.Tensor(param.bs))
        self.assertTrue(val.size()==(param.bs,param.N))

    def test_read_content_weighting(self):
        # two locations, width of three
        memory=self.overwrite_memory()
        read_keys=torch.Tensor(param.bs,param.W,param.R)
        key_strengths=torch.Tensor(param.bs,param.R)
        rcw=memory.read_content_weighting(read_keys,key_strengths)
        self.assertTrue(rcw.size()==(param.bs, param.N, param.R))

    def test_forward_backward_weighting(self):
        memory=self.overwrite_memory()
        bw=memory.backward_weighting()
        fw=memory.forward_weighting()
        self.assertTrue(fw.size()==(param.bs, param.N, param.R))
        self.assertTrue(bw.size()==(param.bs, param.N, param.R))

    def test_read_weighting(self):
        memory=self.overwrite_memory()
        fw=memory.forward_weighting()
        bw=memory.backward_weighting()
        read_keys=torch.Tensor(param.bs,param.W,param.R).normal_()
        read_key_strengths=torch.Tensor(param.bs,param.R).normal_()
        read_modes=torch.Tensor(param.bs,param.R,3).normal_()
        rw=memory.read_weightings(fw, bw, read_keys, read_key_strengths, read_modes)
        self.assertTrue(rw.size()==(param.bs,param.N,param.R))

    def test_retention(self):
        memory=self.overwrite_memory()
        free_gate=torch.Tensor(param.bs,param.R)
        ret=memory.memory_retention(free_gate)
        self.assertTrue(ret.size()==(param.bs, param.N))

    def test_retenion_usage_allocation_flow(self):
        memory=self.overwrite_memory()
        free_gate=torch.Tensor(param.bs,param.R).normal_()
        memory_retention=memory.memory_retention(free_gate)
        write_weighting=torch.Tensor(param.N).normal_()
        memory.update_usage_vector(write_weighting,memory_retention)
        aw=memory.allocation_weighting()
        self.assertTrue(aw.size()==(param.N,))

    def test_write_weighting(self):
        memory=self.overwrite_memory()
        write_key=torch.Tensor([1,2,3])
        write_strength=0.5
        allocation_gate=0.56
        write_gate=0.47
        allocation_weighting=memory.write_content_weighting(write_key,write_strength)
        write_weighting=memory.write_weighting(write_key,write_strength,allocation_gate,
                                  write_gate,allocation_weighting)
        self.assertTrue(write_weighting.size()==(param.N,))

    def test_write_to_memory(self):
        memory=self.overwrite_memory()
        write_key=torch.Tensor([1,2,3])
        allocation_gate=0.56
        write_gate=0.47
        allocation_weighting=memory.write_content_weighting(write_key,write_strength)
        write_weighting=memory.write_weighting(write_key,write_strength,allocation_gate,
                                  write_gate,allocation_weighting)
        write_vector=torch.Tensor(param.W).normal_()
        erase_vector=torch.Tensor(param.W).normal_()
        memory.write_to_memory(write_weighting,erase_vector,write_vector)
        self.assertTrue(memory.memory.size()==(param.N,param.W))

    def test_controller_interface_memory_flow(self):
        ctrl=Controller()
        ctrl.reset_parameters()
        test_input=torch.Tensor(param.x).normal_()
        output,interface_vector=ctrl(test_input)

        interface=Interface()
        ret=interface(interface_vector)
        read_keys, read_strengths, write_key, write_strength, \
        erase_vector, write_vector, free_gates, allocation_gate, \
        write_gate, read_modes=ret

        memory=Memory()
        memory.reset_parameters()

        memory(read_keys, read_strengths, write_key, write_strength,
               erase_vector, write_vector, free_gates, allocation_gate,
               write_gate, read_modes)
        print("controller, interface, memory flow is done")
