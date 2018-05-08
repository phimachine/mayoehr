import unittest
from archi.param import *
from archi.controller import *
import torch

class testRNNUnit(unittest.TestCase):

    def test_RNN_Unit_forward_necessary(self):
        test_input=torch.Tensor(param.bs,param.x+param.R*param.W)
        previous_time=torch.Tensor(param.bs,param.h)
        previous_layer=torch.Tensor(param.bs,param.h)

        ru=RNN_Unit()
        newhid=ru(test_input,previous_time,previous_layer)
        self.assertTrue(newhid.size()==(param.bs,param.h))

class testController(unittest.TestCase):

    def test_controller_forward_necessary(self):
        ctrl=Controller()
        ctrl.reset_parameters()
        # only raw input and the memory read values.
        test_input=torch.Tensor(torch.Tensor(16,param.x+param.R*param.W))
        output,interface=ctrl(test_input)

        self.assertTrue(output.size()==(param.bs,param.v_t))
        self.assertTrue(interface.size()==(param.bs,param.W*param.R+3*param.W+5*param.R+3))