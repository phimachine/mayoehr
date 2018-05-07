import unittest
from archi.param import *
from archi.controller import *
import torch

class testRNNUnit(unittest.TestCase):

    def test_RNN_Unit_forward_necessary(self):
        test_input=torch.Tensor(16,10)
        previous_time=torch.Tensor(16,10)
        previous_layer=torch.Tensor(16,10)

        ru=RNN_Unit()
        newhid=ru(test_input,previous_time,previous_layer)
        self.assertTrue(newhid.size()==(16,10))

class testController(unittest.TestCase):

    def test_controller_forward_necessary(self):
        ctrl=Controller()
        ctrl.reset_parameters()
        test_input=torch.Tensor(torch.Tensor(16,10))
        output,interface=ctrl(test_input)

        self.assertTrue(output.size()==(param.bs,param.v_t))
        self.assertTrue(interface.size()==(param.bs,param.W*param.R+3*param.W+5*param.R+3))