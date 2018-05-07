import unittest
from archi_old.param import *
from archi_old.controller import *
import torch

class testRNNUnit(unittest.TestCase):

    def test_RNN_Unit_forward_necessary(self):
        test_input=torch.Tensor([1,2,3,4,5,6,7,8,9,10])
        previous_time=torch.Tensor([1,2,3,4,5,6,7,8,9,10])
        previous_layer=torch.Tensor([1,2,3,4,5,6,7,8,9,10])

        ru=RNN_Unit()
        newhid=ru(test_input,previous_time,previous_layer)
        self.assertTrue(newhid.size()[0]==10)

class testController(unittest.TestCase):

    def test_controller_forward_necessary(self):
        ctrl=Controller()
        ctrl.reset_parameters()
        test_input=torch.Tensor([1,2,3,4,5,6,7,8,9,10])
        output,interface=ctrl(test_input)

        self.assertTrue(output.size()[0]==param.v_t)
        self.assertTrue(interface.size()[0]==param.W*param.R+3*param.W+5*param.R+3)