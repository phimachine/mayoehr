import unittest
from archi.param import *
from archi.controller import *
import torch

class testRNNUnit(unittest.TestCase):

    def test_RNN_Unit_forward(self):
        test_input=torch.Tensor([1,2,3,4,5,6,7,8,9,10])
        previous_time=torch.Tensor([1,2,3,4,5,6,7,8,9,10])
        previous_layer=torch.Tensor([1,2,3,4,5,6,7,8,9,10])

        ru=RNN_Unit()
        ru(test_input,previous_time,previous_layer)

class testController(unittest.TestCase):

    def test_controller_forward(self):
        ctrl=Controller()
        ctrl.reset_parameters()
        test_input=torch.Tensor([1,2,3,4,5,6,7,8,9,10])
        return ctrl(test_input)