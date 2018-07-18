import unittest
from archi.computer import Computer
import torch
import archi.param as param

class Test_Computer_Necessary(unittest.TestCase):
    def test_computer(self):
        computer=Computer()
        test_input=torch.Tensor(param.bs,param.x)
        output=computer(test_input)
        self.assertTrue(output.size()==(param.bs,param.v_t))