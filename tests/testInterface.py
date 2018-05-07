import unittest
from archi import param
from archi.interface import Interface
from archi.controller import Controller
import torch
from torch.autograd import Variable
import math
from torch.nn.modules import LSTM

class test_interface(unittest.TestCase):

    def test_forward_necessary(self):
        # tests some necessary but insufficient range constraints
        interface_dimension_count=param.W*param.R+param.R+param.W+1+param.W+\
                                  param.W+param.R+1+1+param.R*3
        # must warp everything into a variable. even if gradient is not required
        interface_vector=torch.Tensor(param.bs,interface_dimension_count)
        stdv=1.0/math.sqrt(interface_dimension_count)
        interface_vector.data.uniform_(-stdv,stdv)
        interface=Interface()
        ret=interface(interface_vector)
        read_keys, read_strengths, write_key, write_strength, \
        erase_vector, write_vector, free_gates, allocation_gate, \
        write_gate, read_modes=ret
        self.assertTrue(read_keys.size()==(param.bs,param.W, param.R))
        self.assertTrue(read_strengths.size()==(param.bs,param.R,))
        self.assertTrue((read_strengths>=1).all())
        self.assertTrue(write_key.size()==(param.bs,param.W,))
        self.assertTrue((write_strength>1).all())
        self.assertTrue(erase_vector.size()==(param.bs,param.W,))
        self.assertTrue((erase_vector>0).all())
        self.assertTrue((erase_vector<1).all())
        self.assertTrue(write_vector.size()==(param.bs,param.W,))
        self.assertTrue(free_gates.size()==(param.bs,param.R,))
        self.assertTrue((free_gates>0).all())
        self.assertTrue((free_gates<1).all())
        self.assertTrue((allocation_gate>0).all())
        self.assertTrue((allocation_gate<1).all())
        self.assertTrue((write_gate>0).all())
        self.assertTrue((write_gate<1).all())
        self.assertTrue(read_modes.size()==(param.bs,param.R,3))

    def test_controller_to_interface(self):
        # wires controller to interface and see if it works.

        interface_dimension_count=param.W*param.R+param.R+param.W+1+param.W+\
                                  param.W+param.R+1+1+param.R
        ctrl=Controller()
        ctrl.reset_parameters()
        test_input=torch.Tensor(16,10)
        output,interface_vector=ctrl(test_input)

        interface=Interface()
        ret=interface(interface_vector)
        read_keys, read_strengths, write_key, write_strength, \
        erase_vector, write_vector, free_gates, allocation_gate, \
        write_gate, read_modes=ret

        self.assertTrue(read_keys.size() == (param.bs, param.W, param.R))
        self.assertTrue(read_strengths.size() == (param.bs, param.R,))
        self.assertTrue((read_strengths >= 1).all())
        self.assertTrue(write_key.size() == (param.bs, param.W,))
        self.assertTrue((write_strength > 1).all())
        self.assertTrue(erase_vector.size() == (param.bs, param.W,))
        self.assertTrue((erase_vector > 0).all())
        self.assertTrue((erase_vector < 1).all())
        self.assertTrue(write_vector.size() == (param.bs, param.W,))
        self.assertTrue(free_gates.size() == (param.bs, param.R,))
        self.assertTrue((free_gates > 0).all())
        self.assertTrue((free_gates < 1).all())
        self.assertTrue((allocation_gate > 0).all())
        self.assertTrue((allocation_gate < 1).all())
        self.assertTrue((write_gate > 0).all())
        self.assertTrue((write_gate < 1).all())
        self.assertTrue(read_modes.size() == (param.bs, param.R, 3))